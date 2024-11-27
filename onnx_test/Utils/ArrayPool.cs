using System.Buffers;
using System.Collections.Concurrent;

namespace onnx_test.Utils
{
    public class ImageArrayPool
    {
        private static readonly Lazy<ImageArrayPool> _instance = new(() => new ImageArrayPool());
        public static ImageArrayPool Instance => _instance.Value;

        private readonly ArrayPool<byte> _arrayPool;
        private readonly ConcurrentDictionary<int, Queue<MemoryStream>> _streamPool;
        private readonly int _maxArrayLength;

        private ImageArrayPool(int maxArrayLength = 1024 * 1024 * 10) // 默认最大10MB
        {
            _maxArrayLength = maxArrayLength;
            _arrayPool = ArrayPool<byte>.Create(maxArrayLength, 50); // 最多50个数组在池中
            _streamPool = new ConcurrentDictionary<int, Queue<MemoryStream>>();
        }

        public byte[] RentArray(int minimumLength)
        {
            if (minimumLength > _maxArrayLength)
            {
                throw new ArgumentException($"请求的数组大小({minimumLength})超过了池的最大限制({_maxArrayLength})");
            }
            return _arrayPool.Rent(minimumLength);
        }

        public void ReturnArray(byte[] array)
        {
            if (array != null)
            {
                _arrayPool.Return(array, clearArray: true);
            }
        }

        public MemoryStream RentStream()
        {
            var capacity = 81920; // 默认80KB初始容量
            if (!_streamPool.TryGetValue(capacity, out var queue))
            {
                queue = new Queue<MemoryStream>();
                _streamPool[capacity] = queue;
            }

            lock (queue)
            {
                if (queue.Count > 0)
                {
                    var stream = queue.Dequeue();
                    stream.SetLength(0);
                    stream.Position = 0;
                    return stream;
                }
            }

            return new MemoryStream(capacity);
        }

        public void ReturnStream(MemoryStream stream)
        {
            if (stream == null) return;

            var capacity = (int)stream.Capacity;
            if (!_streamPool.TryGetValue(capacity, out var queue))
            {
                queue = new Queue<MemoryStream>();
                _streamPool[capacity] = queue;
            }

            stream.SetLength(0);
            lock (queue)
            {
                if (queue.Count < 20) // 限制每个容量最多保留20个流
                {
                    queue.Enqueue(stream);
                }
            }
        }
    }
}
