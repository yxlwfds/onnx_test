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

namespace TestNugetCpuOnnx
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("Usage: program <stream_name> <config_file_path>");
                return;
            }

            string streamName = args[0];
            string configPath = args[1];

            if (!File.Exists(configPath))
            {
                Console.WriteLine($"Error: Configuration file not found at path: {configPath}");
                return;
            }

            Yolov5Onnx? yolov5Onnx = null;

            try
            {
                Console.WriteLine("配置文件路径:" + configPath);

                // Initialize model and GPU resources
                ModelManager.Initialize(configPath);

                // Create Yolov5Onnx instance using the initialized model
                yolov5Onnx = new(configPath);

                string lastId = "0-0";
                DateTime lastActive = DateTime.Now;
                MCvScalar color = new(255, 0, 0);

                // Warm up the model
                Console.WriteLine("模型预热...");
                using (var warmupMat = new Mat(640, 640, Emgu.CV.CvEnum.DepthType.Cv8U, 3))
                {
                    yolov5Onnx.DetectLetItRot(warmupMat);
                }

                // 预分配可重用的 Mat
                using (var frame = new Mat(480, 640, DepthType.Cv8U, 3))
                using (var dataSender = new DataSender(streamName))
                {

                    Console.WriteLine("开始从Redis流中读取帧...");
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
                            Console.WriteLine($"Stream {streamName} is offline. Waiting...");
                            await Task.Delay(5000);
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
                                    Console.WriteLine($"图像字节数: {imageBytes.Length} 字节");
                                    // Console.WriteLine($"前10个字节: {BitConverter.ToString(imageBytes.Take(10).ToArray())}");

                                    try
                                    {
                                        Marshal.Copy(imageBytes, 0, frame.DataPointer, imageBytes.Length);
                                        if (!frame.IsEmpty)
                                        {
                                            Console.WriteLine($"图像尺寸: {frame.Size}");
                                            // Perform inference
                                            var stopwatch = Stopwatch.StartNew();
                                            var detectionResult = yolov5Onnx.DetectLetItRot(frame);
                                            stopwatch.Stop();

                                            // Add null check for detectionResult
                                            if (detectionResult == null)
                                            {
                                                Console.WriteLine("检测结果为空");
                                                continue;  // Skip to next iteration
                                            }

                                            Console.WriteLine($"推理完成. 耗时: {stopwatch.ElapsedMilliseconds}毫秒");

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
                                                        dataSender.ProcessBox(streamName, detections, processedImage);

                                                        Console.WriteLine($"检测到的物体数量: {detections.GetLength(0)}");

                                                        // 处理每个检测框的数据
                                                        for (int i = 0; i < detections.GetLength(0); i++)
                                                        {
                                                            float x1 = detections[i, 0];
                                                            float y1 = detections[i, 1];
                                                            float x2 = detections[i, 2];
                                                            float y2 = detections[i, 3];
                                                            float confidence = detections[i, 4];
                                                            int classId = (int)detections[i, 5];

                                                            Console.WriteLine($"检测框 {i + 1}:");
                                                            Console.WriteLine($"  位置: ({x1}, {y1}) - ({x2}, {y2})");
                                                            Console.WriteLine($"  置信度: {confidence:F2}");
                                                            Console.WriteLine($"  类别ID: {classId}");
                                                        }
                                                    }
                                                }
                                            }

                                            // 显示处理后的图片
                                            // CvInvoke.Imshow("Processed Frame", processedImage);
                                            // CvInvoke.WaitKey(1);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"处理图像时出错: {ex.Message}");
                                    }
                                }
                            }
                        }

                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"发生错误: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
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
