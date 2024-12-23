using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;

namespace BingLing.Yolov5Onnx.Gpu
{
    public class ModelManager
    {
        private static InferenceSession? _inferenceSession;
        private static string? _modelPath;
        private static float _iou;
        private static float _confidence;

        public static InferenceSession InferenceSession => _inferenceSession ?? throw new InvalidOperationException("Model not initialized");
        public static float IOU => _iou;
        public static float Confidence => _confidence;

        public static void Initialize(string configPath)
        {
            if (_inferenceSession != null)
                return;

            var config = JsonConvert.DeserializeObject<Dictionary<string, string>>(File.ReadAllText(configPath))
                ?? throw new InvalidOperationException("Failed to load config file");

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _modelPath = Path.Combine(baseDir, config["yolo_onnx_path"]);
            
            if (!float.TryParse(config["iou"], out _iou))
                throw new InvalidOperationException("Invalid IOU value in config");
            
            if (!float.TryParse(config["confidence"], out _confidence))
                throw new InvalidOperationException("Invalid confidence value in config");

            // Initialize GPU session
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableMemoryPattern = true;
            sessionOptions.EnableProfiling = false;
            sessionOptions.EnableCpuMemArena = false;
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = 1;

            // Enable FP16 execution
            sessionOptions.AppendExecutionProvider_CUDA();
            
            // Set provider options after adding the provider
            // sessionOptions.AddSessionConfigEntry("session.use_fp16", "1");
            
            _inferenceSession = new InferenceSession(_modelPath, sessionOptions);
        }

        public static void Cleanup()
        {
            _inferenceSession?.Dispose();
            _inferenceSession = null;
        }
    }
}
