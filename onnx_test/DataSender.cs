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
using onnx_test.Utils;
using System.Runtime.InteropServices;

namespace onnx_test
{
    public class DataSender : IDisposable
    {
        private readonly string _streamName;
        private readonly IDatabase _redisDb;
        private bool _disposed;
        private readonly SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder _jpegEncoder;

        public DataSender(string streamName)
        {
            _streamName = streamName;
            _redisDb = RedisConnectionManager.Instance.GetDatabase();
            _jpegEncoder = new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder
            {
                Quality = 80,
                Interleaved = true
            };
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // 释放托管资源
                    // JpegEncoder from ImageSharp doesn't require explicit disposal
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~DataSender()
        {
            Dispose(false);
        }

        public async Task ProcessBox(string id, float[,] detectionBox, Mat processedImage)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(processedImage);
            
            var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
            var stepStopwatch = new System.Diagnostics.Stopwatch();

            // 从内存池租用数组和流
            byte[] rawData = null;
            byte[] imageBytes = null;
            MemoryStream encoderStream = null;
            MemoryStream base64Stream = null;

            try
            {
                stepStopwatch.Restart();
                var boxList = ProcessDetectionBox(detectionBox);
                Console.WriteLine($"[性能日志] 数据结构转换耗时: {stepStopwatch.ElapsedMilliseconds}ms");

                // 图像编码
                stepStopwatch.Restart();
                
                // 计算所需的缓冲区大小
                int bufferSize = processedImage.Height * processedImage.Width * processedImage.NumberOfChannels;
                rawData = ImageArrayPool.Instance.RentArray(bufferSize);

                // 直接从Mat获取数据，避免额外的复制
                if (processedImage.IsContinuous)
                {
                    Marshal.Copy(processedImage.DataPointer, rawData, 0, bufferSize);
                }
                else
                {
                    using var matData = processedImage.GetUMat(AccessType.Read);
                    matData.CopyTo(rawData);
                }

                // 使用内存池中的流进行编码
                encoderStream = ImageArrayPool.Instance.RentStream();
                using (var image = Image.LoadPixelData<Rgb24>(rawData, processedImage.Width, processedImage.Height))
                {
                    await image.SaveAsync(encoderStream, _jpegEncoder);
                    imageBytes = new byte[encoderStream.Position];
                    encoderStream.Position = 0;
                    await encoderStream.ReadAsync(imageBytes);
                }
                
                Console.WriteLine($"[性能日志] 图像编码耗时: {stepStopwatch.ElapsedMilliseconds}ms");
                Console.WriteLine($"[性能日志] 图像大小: {imageBytes.Length / 1024.0:F2}KB");

                // 发送数据到Redis
                await ProcessDataWithRetry(async () =>
                {
                    await SendToRedisStream(imageBytes, stepStopwatch);
                    await SendToRedisList(id, boxList, imageBytes, stepStopwatch);
                });

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
                // 归还所有租用的资源
                if (rawData != null) ImageArrayPool.Instance.ReturnArray(rawData);
                if (encoderStream != null) ImageArrayPool.Instance.ReturnStream(encoderStream);
                if (base64Stream != null) ImageArrayPool.Instance.ReturnStream(base64Stream);
            }
        }

        private List<List<float>> ProcessDetectionBox(float[,] detectionBox)
        {
            var boxList = new List<List<float>>();
            int rows = detectionBox.GetLength(0);
            int cols = detectionBox.GetLength(1);
            
            for (int i = 0; i < rows; i++)
            {
                var row = new List<float>(cols);
                for (int j = 0; j < cols; j++)
                {
                    row.Add(detectionBox[i, j]);
                }
                boxList.Add(row);
            }
            return boxList;
        }

        private async Task SendToRedisStream(byte[] imageBytes, System.Diagnostics.Stopwatch stepStopwatch)
        {
            stepStopwatch.Restart();
            var streamEntries = new NameValueEntry[]
            {
                new("frame", imageBytes),
                new("timestamp", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString())
            };
            await _redisDb.StreamAddAsync(_streamName, streamEntries, maxLength: 50);
            Console.WriteLine($"[性能日志] Redis Stream写入耗时: {stepStopwatch.ElapsedMilliseconds}ms");
        }

        private async Task SendToRedisList(string id, List<List<float>> boxList, byte[] imageBytes, System.Diagnostics.Stopwatch stepStopwatch)
        {
            stepStopwatch.Restart();
            using var base64Stream = ImageArrayPool.Instance.RentStream();
            using (var writer = new StreamWriter(base64Stream, leaveOpen: true))
            {
                writer.Write(Convert.ToBase64String(imageBytes));
                writer.Flush();
                base64Stream.Position = 0;
                
                var dataOut = new
                {
                    id = id,
                    res = JsonSerializer.Serialize(boxList),
                    image = new StreamReader(base64Stream).ReadToEnd()
                };

                string jsonData = JsonSerializer.Serialize(dataOut);
                await _redisDb.ListLeftPushAsync("box_data_queue", jsonData);
                await _redisDb.ListTrimAsync("box_data_queue", 0, 1000);
            }
            Console.WriteLine($"[性能日志] Redis操作耗时: {stepStopwatch.ElapsedMilliseconds}ms");
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
    }
}
