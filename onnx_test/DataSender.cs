using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using StackExchange.Redis;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util; 
using System.Drawing;
using System.Runtime.InteropServices;

namespace onnx_test
{
    public class DataSender : IDisposable
    {
        private readonly ConcurrentQueue<RawBoxData> _sendQueue;
        private readonly CancellationTokenSource _cancellationTokenSource;
        private Task _processingTask;
        private bool _disposed;
        private readonly AutoResetEvent _queueNotification;
        private readonly string _streamName;
        private readonly IDatabase _redisDb;

        private class RawBoxData : IDisposable
        {
            public string Id { get; set; }
            public float[,] DetectionBox { get; set; }
            public Mat Image { get; set; }

            public void Dispose()
            {
                Image?.Dispose();
            }
        }

        public DataSender(string streamName)
        {
            _sendQueue = new ConcurrentQueue<RawBoxData>();
            _cancellationTokenSource = new CancellationTokenSource();
            _queueNotification = new AutoResetEvent(false);
            _streamName = streamName;
            _redisDb = RedisConnectionManager.Instance.GetDatabase();
        }

        public void ProcessBox(string id, float[,] numBox, Mat outImg)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            try
            {
                var rawData = new RawBoxData
                {
                    Id = id,
                    DetectionBox = numBox,
                    Image = outImg.Clone()
                };

                _sendQueue.Enqueue(rawData);
                _queueNotification.Set(); // 通知处理线程有新数据
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
                    await Task.Delay(1000 * (i + 1)); // 指数退避
                }
            }
        }

        public void StartBoxService()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            _processingTask = Task.Run(async () =>
            {
                try
                {
                    while (!_cancellationTokenSource.Token.IsCancellationRequested)
                    {
                        await Task.Delay(1);
                        // 等待新数据或取消信号
                        WaitHandle.WaitAny(new[] { _queueNotification, _cancellationTokenSource.Token.WaitHandle });

                        // 处理队列中的所有数据
                        while (_sendQueue.TryDequeue(out RawBoxData rawData))
                        {
                            using (rawData) // 自动处理Mat的释放
                            {
                                try
                                {
                                    var boxList = new List<List<float>>();
                                    int rows = rawData.DetectionBox.GetLength(0);
                                    int cols = rawData.DetectionBox.GetLength(1);
                                    for (int i = 0; i < rows; i++)
                                    {
                                        var row = new List<float>();
                                        for (int j = 0; j < cols; j++)
                                        {
                                            row.Add(rawData.DetectionBox[i, j]);
                                        }
                                        boxList.Add(row);
                                    }

                                    byte[] imageBytes;
                                    using (var vectorByte = new VectorOfByte())
                                    {
                                        CvInvoke.Imencode(".png", rawData.Image, vectorByte,
                                        [
                                            new(ImwriteFlags.PngCompression, 9) 
                                        ]);
                                        imageBytes = vectorByte.ToArray();
                                    }

                                    // Send image data to Redis stream with retry
                                    await ProcessDataWithRetry(async () =>
                                    {
                                        var streamEntries = new NameValueEntry[]
                                        {
                                            new("frame", imageBytes),
                                            new("timestamp", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString())
                                        };
                                        await _redisDb.StreamAddAsync(_streamName, streamEntries, maxLength: 50);

                                        string imgBase64 = Convert.ToBase64String(imageBytes);
                                        var dataOut = new
                                        {
                                            id = rawData.Id,
                                            res = JsonSerializer.Serialize(boxList),
                                            image = imgBase64
                                        };

                                        string jsonData = JsonSerializer.Serialize(dataOut);
                                        Console.WriteLine(dataOut.res);
                                        await _redisDb.ListLeftPushAsync("box_data_queue", jsonData);
                                        await _redisDb.ListTrimAsync("box_data_queue", 0, 1000);
                                    });
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"Error processing data: {ex.Message}");
                                    if (ex is RedisConnectionException)
                                    {
                                        Console.WriteLine("Redis connection error, will retry with next data");
                                    }
                                }
                            }
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation is requested
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Fatal error in processing queue: {ex.Message}");
                    throw;
                }
            });
        }

        public async Task StopAsync()
        {
            if (_cancellationTokenSource != null && !_disposed)
            {
                _cancellationTokenSource.Cancel();
                if (_processingTask != null)
                {
                    try
                    {
                        await _processingTask;
                    }
                    catch (OperationCanceledException)
                    {
                        // Console.WriteLine("Processing task canceled.");
                    }
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _cancellationTokenSource?.Cancel();
                    if (_processingTask != null)
                    {
                        try
                        {
                            _processingTask.Wait(TimeSpan.FromSeconds(5));
                        }
                        catch (Exception ex) when (ex is OperationCanceledException || ex is AggregateException)
                        {
                            // Expected when task is cancelled
                        }
                    }
                    _queueNotification?.Dispose();
                    
                    // Clear and dispose any remaining items in the queue
                    while (_sendQueue.TryDequeue(out RawBoxData item))
                    {
                        item?.Dispose();
                    }
                }
                _disposed = true;
            }
        }

        ~DataSender()
        {
            Dispose(false);
        }
    }
}
