using NLog;

namespace onnx_test
{
    public sealed class ExceptionHandler
    {
        private static readonly Lazy<ExceptionHandler> _instance = new(() => new ExceptionHandler());
        private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();
        private static readonly object _lockObject = new();
        private bool _isInitialized;

        private ExceptionHandler() { }
        public static ExceptionHandler Instance => _instance.Value;

        public void Initialize()
        {
            if (_isInitialized) return;

            lock (_lockObject)
            {
                if (_isInitialized) return;

                AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
                {
                    var exception = args.ExceptionObject as Exception;
                    _logger.Fatal(exception, $"Unhandled exception: {exception?.Message}");
                };

                TaskScheduler.UnobservedTaskException += (sender, args) =>
                {
                    _logger.Fatal(args.Exception, $"Unobserved task exception: {args.Exception.Message}");
                    args.SetObserved();
                };

                _isInitialized = true;
            }
        }

        public void LogException(Exception? ex, string context, bool rethrow = true)
        {
            if (ex == null) return;

            try
            {
                _logger.Error(ex, $"[{context}] {ex.Message}");
                
                if (ex.InnerException != null)
                {
                    _logger.Error(ex.InnerException, $"[{context}] Inner Exception: {ex.InnerException.Message}");
                }
            }
            catch
            {
                // Prevent logging errors from causing additional problems
            }

            if (rethrow)
            {
                throw ex;
            }
        }
    }
}
