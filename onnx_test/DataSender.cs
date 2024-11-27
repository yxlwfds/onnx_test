using System;
using System.Collections.Generic;
using System.Text.Json;
using StackExchange.Redis;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

namespace onnx_test
{
    public class DataSender : IDisposable
    {
        private readonly string _streamName;
        private readonly IDatabase _redisDb;
        private bool _disposed;

        public DataSender(string streamName)
        {
            _streamName = streamName;
            _redisDb = RedisConnectionManager.Instance.GetDatabase();
        }

        public async Task ProcessBox(string id, float[,] detectionBox, Mat processedImage)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
            var stepStopwatch = new System.Diagnostics.Stopwatch();

            try
            {
                stepStopwatch.Restart();
                // Convert detection box to list format
                var boxList = new List<List<float>>();
                int rows = detectionBox.GetLength(0);
                int cols = detectionBox.GetLength(1);
                for (int i = 0; i < rows; i++)
                {
                    var row = new List<float>();
                    for (int j = 0; j < cols; j++)
                    {
                        row.Add(detectionBox[i, j]);
                    }
                    boxList.Add(row);
                }
                Console.WriteLine($"[性能日志] 数据结构转换耗时: {stepStopwatch.ElapsedMilliseconds}ms");

                // Encode image to JPEG bytes
                stepStopwatch.Restart();
                byte[] imageBytes;
                
                // 从Mat获取原始图像数据并立即释放
                byte[] rawData;
                using (var matData = processedImage.GetUMat(Emgu.CV.CvEnum.AccessType.Read))
                {
                    rawData = new byte[processedImage.Height * processedImage.Width * processedImage.NumberOfChannels];
                    matData.CopyTo(rawData);
                }

                // 使用ImageSharp进行高性能编码，确保所有资源都被释放
                using (var image = Image.LoadPixelData<Rgb24>(rawData, processedImage.Width, processedImage.Height))
                using (var ms = new MemoryStream())
                {
                    var encoder = new JpegEncoder
                    {
                        Quality = 80,
                        Interleaved = true
                    };
                    
                    image.Save(ms, encoder);
                    imageBytes = ms.ToArray();
                }
                
                // 清理不再需要的数据
                Array.Clear(rawData, 0, rawData.Length);
                rawData = null;
                
                Console.WriteLine($"[性能日志] 图像编码耗时: {stepStopwatch.ElapsedMilliseconds}ms");
                Console.WriteLine($"[性能日志] 图像大小: {imageBytes.Length / 1024.0:F2}KB");

                // Send data to Redis with retry
                await ProcessDataWithRetry(async () =>
                {
                    stepStopwatch.Restart();
                    // Send image to Redis stream
                    var streamEntries = new NameValueEntry[]
                    {
                        new("frame", imageBytes),
                        new("timestamp", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString())
                    };
                    await _redisDb.StreamAddAsync(_streamName, streamEntries, maxLength: 50);
                    Console.WriteLine($"[性能日志] Redis Stream写入耗时: {stepStopwatch.ElapsedMilliseconds}ms");

                    stepStopwatch.Restart();
                    // Send detection results and convert to Base64 in chunks to reduce memory usage
                    using (var ms = new MemoryStream())
                    {
                        var writer = new StreamWriter(ms);
                        writer.Write(Convert.ToBase64String(imageBytes));
                        writer.Flush();
                        ms.Position = 0;
                        
                        var dataOut = new
                        {
                            id = id,
                            res = JsonSerializer.Serialize(boxList),
                            image = new StreamReader(ms).ReadToEnd()
                        };
                        Console.WriteLine($"[性能日志] Base64编码耗时: {stepStopwatch.ElapsedMilliseconds}ms");

                        stepStopwatch.Restart();
                        string jsonData = JsonSerializer.Serialize(dataOut);
                        Console.WriteLine(dataOut.res);
                        await _redisDb.ListLeftPushAsync("box_data_queue", jsonData);
                        await _redisDb.ListTrimAsync("box_data_queue", 0, 1000);
                        Console.WriteLine($"[性能日志] Redis List操作耗时: {stepStopwatch.ElapsedMilliseconds}ms");
                    }
                });

                // 清理图像数据
                Array.Clear(imageBytes, 0, imageBytes.Length);
                imageBytes = null;

                totalStopwatch.Stop();
                Console.WriteLine($"[性能日志] ProcessBox总耗时: {totalStopwatch.ElapsedMilliseconds}ms");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in ProcessBox: {ex.Message}");
                throw;
            }
            finally
            {
                // 确保在发生异常时也能清理资源
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        private static async Task ProcessDataWithRetry(Func<Task> action, int maxRetries = 3)
        {
            for (int i = 0; i < maxRetries; i++)
            {
                try
                {
                    await action();
                    return;
                }
                catch (RedisConnectionException)
                {
                    if (i == maxRetries - 1) throw;
                    await Task.Delay(1000 * (i + 1));
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
}
