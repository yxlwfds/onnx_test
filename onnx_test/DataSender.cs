using System;
using System.Collections.Generic;
using System.Text.Json;
using StackExchange.Redis;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

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
                using (var vectorByte = new VectorOfByte())
                {
                    // 使用JPEG格式，质量设为80（0-100之间，值越大质量越好，但文件更大）
                    CvInvoke.Imencode(".jpg", processedImage, vectorByte,
                    [
                        new(ImwriteFlags.JpegQuality, 80),
                        new(ImwriteFlags.JpegOptimize, 1)
                    ]);
                    imageBytes = vectorByte.ToArray();
                }
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
                    // Send detection results
                    string imgBase64 = Convert.ToBase64String(imageBytes);
                    var dataOut = new
                    {
                        id = id,
                        res = JsonSerializer.Serialize(boxList),
                        image = imgBase64
                    };
                    Console.WriteLine($"[性能日志] Base64编码耗时: {stepStopwatch.ElapsedMilliseconds}ms");

                    stepStopwatch.Restart();
                    string jsonData = JsonSerializer.Serialize(dataOut);
                    Console.WriteLine(dataOut.res);
                    await _redisDb.ListLeftPushAsync("box_data_queue", jsonData);
                    await _redisDb.ListTrimAsync("box_data_queue", 0, 1000);
                    Console.WriteLine($"[性能日志] Redis List操作耗时: {stepStopwatch.ElapsedMilliseconds}ms");
                });

                totalStopwatch.Stop();
                Console.WriteLine($"[性能日志] ProcessBox总耗时: {totalStopwatch.ElapsedMilliseconds}ms");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in ProcessBox: {ex.Message}");
                throw;
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
