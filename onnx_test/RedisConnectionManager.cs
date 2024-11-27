using StackExchange.Redis;
using System;
using System.Threading.Tasks;

namespace onnx_test
{
    public class RedisConnectionManager : IDisposable
    {
        private static readonly Lazy<RedisConnectionManager> _instance = new(() => new RedisConnectionManager());
        private readonly ConnectionMultiplexer _redis;
        private readonly ConfigurationOptions _configOptions;
        private bool _disposed;

        public static RedisConnectionManager Instance => _instance.Value;

        private RedisConnectionManager()
        {
            _configOptions = new ConfigurationOptions
            {
                EndPoints = { "127.0.0.1:6379" },
                AbortOnConnectFail = false,
                ConnectRetry = 3,
                ConnectTimeout = 5000,
                SyncTimeout = 5000,
                AllowAdmin = true,
                KeepAlive = 60,
                DefaultDatabase = 0
            };

            _redis = ConnectionMultiplexer.Connect(_configOptions);
        }

        public IDatabase GetDatabase(int db = -1) => _redis.GetDatabase(db);

        public async Task<IDatabase> GetDatabaseAsync(int db = -1)
        {
            await Task.CompletedTask; // Ensure async context
            return _redis.GetDatabase(db);
        }

        public async Task<bool> StringSetAsync(string key, string value, TimeSpan? expiry = null)
        {
            var db = GetDatabase();
            return await db.StringSetAsync(key, value, expiry);
        }

        public async Task<RedisValue> StringGetAsync(string key)
        {
            var db = GetDatabase();
            return await db.StringGetAsync(key);
        }

        public async Task<long> ListLeftPushAsync(string key, string value)
        {
            var db = GetDatabase();
            return await db.ListLeftPushAsync(key, value);
        }

        public async Task ListTrimAsync(string key, long start, long stop)
        {
            var db = GetDatabase();
            await db.ListTrimAsync(key, start, stop);
        }

        public async Task<bool> HashSetAsync(string key, string hashField, string value)
        {
            var db = GetDatabase();
            return await db.HashSetAsync(key, hashField, value);
        }

        public async Task<RedisValue> HashGetAsync(string key, string hashField)
        {
            var db = GetDatabase();
            return await db.HashGetAsync(key, hashField);
        }

        public async Task<bool> KeyDeleteAsync(string key)
        {
            var db = GetDatabase();
            return await db.KeyDeleteAsync(key);
        }

        public async Task<long> StreamAddAsync(string key, string field, byte[] value, int maxLength = 0)
        {
            var db = GetDatabase();
            var streamEntry = new NameValueEntry[] { new NameValueEntry(field, value) };
            var messageId = await db.StreamAddAsync(key, streamEntry, maxLength: maxLength);
            return (long)messageId; 
        }

        public async Task<StreamEntry[]> StreamReadAsync(string key, string position)
        {
            var db = GetDatabase();
            return await db.StreamReadAsync(key, position);
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
                    _redis?.Dispose();
                }
                _disposed = true;
            }
        }

        ~RedisConnectionManager()
        {
            Dispose(false);
        }
    }
}
