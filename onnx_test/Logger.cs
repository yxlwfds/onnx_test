using NLog;

namespace onnx_test
{
    public sealed class ApplicationLogger
    {
        private static readonly Lazy<ApplicationLogger> _instance = new(() => new ApplicationLogger());
        private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

        private ApplicationLogger() { }
        public static ApplicationLogger Instance => _instance.Value;

        public void Debug(string message)
        {
            if (_logger.IsEnabled(LogLevel.Debug))
                _logger.Debug(message);
        }

        public void Info(string message)
        {
            if (_logger.IsEnabled(LogLevel.Info))
                _logger.Info(message);
        }

        public void Warning(string message)
        {
            if (_logger.IsEnabled(LogLevel.Warn))
                _logger.Warn(message);
        }

        public void Error(string message, Exception? ex = null)
        {
            if (_logger.IsEnabled(LogLevel.Error))
            {
                if (ex != null)
                    _logger.Error(ex, message);
                else
                    _logger.Error(message);
            }
        }

        public void Fatal(string message, Exception? ex = null)
        {
            if (_logger.IsEnabled(LogLevel.Fatal))
            {
                if (ex != null)
                    _logger.Fatal(ex, message);
                else
                    _logger.Fatal(message);
            }
        }
    }
}
