using BingLing.Yolov5Onnx.Gpu;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Collections.Concurrent;
using System.Diagnostics;
using StackExchange.Redis;
using System.Text;
using System.Runtime.InteropServices;

namespace TestNugetCpuOnnx
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            ConnectionMultiplexer? redis = null;
            Yolov5Onnx? yolov5Onnx = null;
            
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string configPath = Path.Combine(baseDir, "yolov5_onnx.json");
                Console.WriteLine("配置文件路径:" + configPath);

                // Initialize model and GPU resources
                ModelManager.Initialize(configPath);

                // Create Yolov5Onnx instance using the initialized model
                yolov5Onnx = new(configPath);

                // Redis connection
                redis = ConnectionMultiplexer.Connect("127.0.0.1:6379");
                var db = redis.GetDatabase();

                string streamName = "stream5"; // 替换为实际的stream名称
                string lastId = "0-0";
                DateTime lastActive = DateTime.Now;
                MCvScalar color = new(255, 0, 0);

                // Warm up the model
                Console.WriteLine("模型预热...");
                using (var warmupMat = new Mat(640, 640, Emgu.CV.CvEnum.DepthType.Cv8U, 3))
                {
                    yolov5Onnx.DetectLetItRot(warmupMat);
                }

                Console.WriteLine("开始从Redis流中读取帧...");
                while (true)
                {
                    var now = DateTime.Now;

                    // Update active status every second
                    if ((now - lastActive).TotalSeconds >= 1)
                    {
                        await db.StringSetAsync($"{streamName}_push_active", now.Ticks.ToString());
                        lastActive = now;
                    }

                    // Check if stream is offline
                    var status = await db.HashGetAsync($"stream_info_{streamName}", "status");
                    if (status == "offline")
                    {
                        Console.WriteLine($"Stream {streamName} is offline. Exiting...");
                        await Task.Delay(5000);
                    }

                    if (status != "offline")
                    {
                        // Read from stream
                        var result = await db.StreamReadAsync($"frame_{streamName}", lastId, 1);

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
                                Console.WriteLine($"前10个字节: {BitConverter.ToString(imageBytes.Take(10).ToArray())}");

                                try
                                {
                                    Console.WriteLine("开始处理图像...");
                                    // Assuming 640x480 RGB format
                                    int width = 640;
                                    int height = 480;
                                    using (Mat frame = new Mat(height, width, DepthType.Cv8U, 3))
                                    {
                                        Marshal.Copy(imageBytes, 0, frame.DataPointer, imageBytes.Length);
                                        Console.WriteLine("图像加载完成");

                                        if (frame != null && !frame.IsEmpty)
                                        {
                                            Console.WriteLine($"图像尺寸: {frame.Size}");
                                            // Perform inference
                                            var stopwatch = Stopwatch.StartNew();
                                            var detectionResult = yolov5Onnx.DetectLetItRot(frame);
                                            stopwatch.Stop();

                                            Console.WriteLine($"推理完成. 耗时: {stopwatch.ElapsedMilliseconds}毫秒");
                                            
                                            // 获取处理后的图片
                                            Mat processedImage = detectionResult.ProcessedImage;
                                            // 获取检测框数据
                                            float[,] detections = detectionResult.Outputs;
                                            
                                            // 打印检测到的物体数量
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
                                            
                                            // 显示处理后的图片
                                            CvInvoke.Imshow("Processed Frame", processedImage);
                                            CvInvoke.WaitKey(1);
                                        }
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"处理图像时出错: {ex.Message}");
                                }
                            }
                        }
                    }

                    await Task.Delay(1); // Small delay to prevent CPU overuse
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
                if (redis != null)
                {
                    redis.Close();
                    redis.Dispose();
                }
                
                if (yolov5Onnx != null && yolov5Onnx is IDisposable disposable)
                {
                    disposable.Dispose();
                }
                
                ModelManager.Cleanup();
            }
        }
    }
}
