using BingLing.Yolov5Onnx.Gpu;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Collections.Concurrent;
using System.Diagnostics;
using StackExchange.Redis;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;
using onnx_test;
using NLog;

using CustomLogger = onnx_test.ApplicationLogger;

namespace TestNugetCpuOnnx
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            // 确保日志目录存在
            Directory.CreateDirectory(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs"));
            
            // 加载NLog配置
            LogManager.LoadConfiguration("nlog.config");
            
            if (args.Length != 2)
            {
                ApplicationLogger.Instance.Error("Usage: program <stream_name> <config_file_path>");
                return;
            }

            string streamName = args[0];
            string configPath = args[1];

            if (!File.Exists(configPath))
            {
                ApplicationLogger.Instance.Error($"Error: Configuration file not found at path: {configPath}");
                return;
            }

            Yolov5Onnx? yolov5Onnx = null;

            try
            {
                ApplicationLogger.Instance.Info("配置文件路径:" + configPath);

                // Initialize model and GPU resources
                ModelManager.Initialize(configPath);

                // Create Yolov5Onnx instance using the initialized model
                yolov5Onnx = new(configPath);

                string lastId = "0-0";
                DateTime lastActive = DateTime.Now;
                MCvScalar color = new(255, 0, 0);

                // Warm up the model
                ApplicationLogger.Instance.Info("模型预热...");
                using (var warmupMat = new Mat(640, 640, Emgu.CV.CvEnum.DepthType.Cv8U, 3))
                {
                    yolov5Onnx.DetectLetItRot(warmupMat);
                }

                // 预分配可重用的 Mat
                using (var frame = new Mat(640, 640, DepthType.Cv8U, 3))
                using (var dataSender = new DataSender(streamName))
                {

                    ApplicationLogger.Instance.Info("开始从Redis流中读取帧...");
                    while (true)
                    {
                        await Task.Delay(1);
                        var now = DateTime.Now;

                        // Update active status every second
                        if ((now - lastActive).TotalSeconds >= 1)
                        {
                            await RedisConnectionManager.Instance.StringSetAsync($"{streamName}_push_active", now.Ticks.ToString());
                            lastActive = now;
                        }

                        // Check if stream is offline
                        var status = await RedisConnectionManager.Instance.HashGetAsync($"stream_info_{streamName}", "status");
                        if (status == "offline")
                        {
                            ApplicationLogger.Instance.Warning($"Stream {streamName} is offline. Waiting...");
                            await Task.Delay(1000);
                        }

                        if (status != "offline")
                        {
                            // Read from stream
                            var result = await RedisConnectionManager.Instance.StreamReadAsync($"frame_no_yolo_{streamName}", lastId);

                            if (result != null && result.Length > 0)
                            {
                                var entry = result[0];
                                lastId = entry.Id;

                                // Get frame data
                                var frameData = entry.Values.FirstOrDefault(x => x.Name == "frame").Value;

                                if (frameData.HasValue)
                                {
                                    // Convert byte array to Mat
                                    byte[] imageBytes = (byte[])frameData;
                                    ApplicationLogger.Instance.Debug($"图像字节数: {imageBytes.Length} 字节");

                                    try
                                    {
                                        var frameProcessingStopwatch = Stopwatch.StartNew();
                                        Marshal.Copy(imageBytes, 0, frame.DataPointer, imageBytes.Length);
                                        if (!frame.IsEmpty)
                                        {
                                            ApplicationLogger.Instance.Debug($"图像尺寸: {frame.Size}");
                                            // Perform inference
                                            var stopwatch = Stopwatch.StartNew();
                                            var detectionResult = yolov5Onnx.DetectLetItRot(frame);
                                            stopwatch.Stop();

                                            // Add null check for detectionResult
                                            if (detectionResult == null)
                                            {
                                                ApplicationLogger.Instance.Warning("检测结果为空");
                                                continue;  // Skip to next iteration
                                            }

                                            ApplicationLogger.Instance.Info($"推理完成. 耗时: {stopwatch.ElapsedMilliseconds}毫秒");

                                            // 获取处理后的图片和检测框数据
                                            using (Mat processedImage = detectionResult.ProcessedImage)
                                            {
                                                float[,] detections = detectionResult.Outputs;

                                                // 打印检测到的物体数量
                                                if (detections != null)
                                                {
                                                    // 只有当检测到物体时才发送数据到队列
                                                    if (detections.Length > 0 && dataSender != null)
                                                    {
                                                       await dataSender.ProcessBox(streamName, detections, processedImage);

                                                        ApplicationLogger.Instance.Info($"检测到的物体数量: {detections.GetLength(0)}");

                                                        // 处理每个检测框的数据
                                                        for (int i = 0; i < detections.GetLength(0); i++)
                                                        {
                                                            float x1 = detections[i, 0];
                                                            float y1 = detections[i, 1];
                                                            float x2 = detections[i, 2];
                                                            float y2 = detections[i, 3];
                                                            float confidence = detections[i, 4];
                                                            int classId = (int)detections[i, 5];

                                                            ApplicationLogger.Instance.Info($"检测框 {i + 1}:");
                                                            ApplicationLogger.Instance.Info($"  位置: ({x1}, {y1}) - ({x2}, {y2})");
                                                            ApplicationLogger.Instance.Info($"  置信度: {confidence:F2}");
                                                            ApplicationLogger.Instance.Info($"  类别ID: {classId}");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        frameProcessingStopwatch.Stop();
                                        ApplicationLogger.Instance.Info($"帧处理总耗时: {frameProcessingStopwatch.ElapsedMilliseconds}毫秒");
                                    }
                                    catch (Exception ex)
                                    {
                                        ApplicationLogger.Instance.Error($"处理图像时出错: {ex.Message}");
                                    }
                                }
                            }
                        }

                    }
                }
            }
            catch (Exception ex)
            {
                ApplicationLogger.Instance.Error($"发生错误: {ex.Message}");
                ApplicationLogger.Instance.Error(ex.StackTrace);
            }
            finally
            {
                // Cleanup all resources
                if (yolov5Onnx != null && yolov5Onnx is IDisposable disposable)
                {
                    disposable.Dispose();
                }

                ModelManager.Cleanup();
            }
        }
    }
}
