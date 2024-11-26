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
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string configPath = Path.Combine(baseDir, "yolov5_onnx.json");
                Console.WriteLine("配置文件路径:" + configPath);

                // Initialize model and GPU resources
                ModelManager.Initialize(configPath);
                
                // Create Yolov5Onnx instance using the initialized model
                Yolov5Onnx yolov5Onnx = new(configPath);

                // Redis connection
                var redis = ConnectionMultiplexer.Connect("127.0.0.1:6379");
                var db = redis.GetDatabase();
                
                string streamName = "stream5"; // 替换为实际的stream名称
                string lastId = "0-0";
                DateTime lastActive = DateTime.Now;
                MCvScalar color = new(255, 0, 0);

                // Warm up the model
                Console.WriteLine("模型预热...");
                using var warmupMat = new Mat(640, 640, Emgu.CV.CvEnum.DepthType.Cv8U, 3);
                yolov5Onnx.DetectLetItRot(warmupMat);

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
                    //print status
                    Console.WriteLine($"流状态: {status}");
                    
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
                            //print frame data
                            // Console.WriteLine($"帧数据: {frameData}");
                            
                            if (frameData.HasValue)
                            {
                                // Convert byte array to Mat
                                byte[] imageBytes = (byte[])frameData;
                                Console.WriteLine($"图像字节数: {imageBytes.Length} 字节");
                                Console.WriteLine($"前10个字节: {BitConverter.ToString(imageBytes.Take(10).ToArray())}");
                                
                                try 
                                {
                                    // Assuming 640x480 RGB format
                                    int width = 640;
                                    int height = 480;
                                    Mat frame = new Mat(height, width, DepthType.Cv8U, 3);
                                    Marshal.Copy(imageBytes, 0, frame.DataPointer, imageBytes.Length);
                                    
                                    Console.WriteLine("开始处理图像...");
                                    Console.WriteLine("图像加载完成");
                                    Console.WriteLine($"帧是否为空: {frame == null}");
                                    Console.WriteLine($"帧是否无数据: {frame.IsEmpty}");
                                    
                                    if (frame != null && !frame.IsEmpty)
                                    {
                                        Console.WriteLine($"图像尺寸: {frame.Size}");
                                        // Perform inference
                                        var stopwatch = Stopwatch.StartNew();
                                        var predictions = yolov5Onnx.DetectLetItRot(frame);
                                        stopwatch.Stop();

                                        Console.WriteLine($"推理完成. 耗时: {stopwatch.ElapsedMilliseconds}毫秒");
                                        Console.WriteLine($"检测到的物体数量: {predictions.Sum(p => p.Value.Count)}");
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
                Console.WriteLine($"错误: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
                ModelManager.Cleanup();
            }
        }
    }
}
