using Emgu.CV;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV.CvEnum;
using onnx_test;

namespace BingLing.Yolov5Onnx.Gpu
{
    /// <summary>
    /// Yolo类
    /// </summary>
    public class Yolov5Onnx : IDisposable
    {
        /// <summary>
        /// yolov5 onnx文件的物理路径
        /// </summary>
        public string? yolo_onnx_path;
        /// <summary>
        /// yolov5 onnx文件的物理路径
        /// </summary>
        public string? YoloOnnxPath
        {
            set
            {
                yolo_onnx_path = value;
            }
            get
            {
                return yolo_onnx_path;
            }
        }

        /// <summary>
        /// 两个预测结果交并比至少是此值的时候，认为这两个结果预测的同一个目标，从而忽视掉置信度较低的一个预测结果
        /// </summary>
        private float iou;
        /// <summary>
        /// 两个预测结果交并比至少是此值的时候，认为这两个结果预测的同一个目标，从而忽视掉置信度较低的一个预测结果
        /// </summary>
        public float IOU
        {
            set
            {
                iou = value;
            }
            get
            {
                return iou;
            }
        }

        /// <summary>
        /// 预测结果置信度至少是此值的时候，才认为这是一个有效结果，否则忽视它
        /// </summary>
        private float confidence;
        /// <summary>
        /// 预测结果置信度至少是此值的时候，才认为这是一个有效结果，否则忽视它
        /// </summary>
        public float Confidence
        {
            set
            {
                confidence = value;
            }
            get
            {
                return confidence;
            }
        }

        private InferenceSession? inference_session;

        /// <summary>
        /// 推理会话，隶属于OnnxRuntime官方类，有关介绍请到官网查询
        /// <see href="https://onnxruntime.ai/docs/"/>
        /// </summary>
        public InferenceSession? InferenceSession
        {
            get
            {
                return inference_session;
            }
        }

        private DenseTensor<float>? _reuseInputTensor;
        private byte[]? _imageBuffer;
        private float[]? _processingBuffer;
        private readonly object _bufferLock = new object();
        private Mat? _resizedMat;
        private readonly Size _modelInputSize;
        private readonly string _inputMetadataName;
        private readonly string _outputMetadataName;
        private readonly int[] _outputDimensions;
        private readonly int _lengthOfPredict;
        private readonly int _countOfKind;

        private bool _disposed = false;

        /// <summary>
        /// 以json配置文件构造Yolo对象，例如
        /// <![CDATA[
        /// {
        ///     "yolo_onnx_path": "yolov5s.onnx",
        ///     "iou": 0.75,
        ///     "confidence": 0.65
        /// }
        /// ]]>
        /// </summary>
        /// <param name="configPath">配置文件物理路径</param>
        public Yolov5Onnx(string configPath)
        {
            var config = JsonConvert.DeserializeObject<Dictionary<string, string>>(File.ReadAllText(configPath));
            if (config == null)
                throw new ArgumentException("Invalid config file");

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            this.yolo_onnx_path = Path.Combine(baseDir, config["yolo_onnx_path"]);
            
            if (!float.TryParse(config["iou"], out this.iou))
                throw new ArgumentException("Invalid IOU value in config");
            
            if (!float.TryParse(config["confidence"], out this.confidence))
                throw new ArgumentException("Invalid confidence value in config");

            ModelManager.Initialize(configPath);
            this.inference_session = ModelManager.InferenceSession;
            
            // Cache metadata names
            _inputMetadataName = inference_session.InputNames[0];
            _outputMetadataName = inference_session.OutputNames[0];
            
            // Pre-allocate model input size
            var dimensions = inference_session.InputMetadata[_inputMetadataName].Dimensions;
            _modelInputSize = new Size(dimensions[3], dimensions[2]);
            
            // Cache output dimensions
            _outputDimensions = inference_session.OutputMetadata[_outputMetadataName].Dimensions;
            _lengthOfPredict = _outputDimensions[1];
            _countOfKind = _outputDimensions[2] - 5;
            
            // Pre-allocate reusable Mat
            _resizedMat = new Mat(_modelInputSize.Height, _modelInputSize.Width, DepthType.Cv8U, 3);
            
            // Pre-allocate buffers
            int maxSize = _modelInputSize.Width * _modelInputSize.Height * 3;
            _imageBuffer = new byte[maxSize];
            _processingBuffer = new float[maxSize];
            _reuseInputTensor = new DenseTensor<float>(dimensions);
        }

        private Mat LetterboxImage(Mat img, Size newShape, Color color, bool auto = true, bool scaleFill = false, bool scaleup = true, int stride = 32)
        {
            float ratio = Math.Min((float)newShape.Width / img.Width, (float)newShape.Height / img.Height);
            if (!scaleup)
                ratio = Math.Min(ratio, 1.0f);

            int newUnpad = (int)Math.Round(img.Width * ratio);
            int newUnpadH = (int)Math.Round(img.Height * ratio);
            int dw = newShape.Width - newUnpad;
            int dh = newShape.Height - newUnpadH;

            if (auto)
            {
                dw = dw % stride;
                dh = dh % stride;
            }

            var resized = new Mat();
            CvInvoke.Resize(img, resized, new Size(newUnpad, newUnpadH));

            int top = dh / 2;
            int bottom = dh - (dh / 2);
            int left = dw / 2;
            int right = dw - (dw / 2);

            var padded = new Mat();
            CvInvoke.CopyMakeBorder(resized, padded, top, bottom, left, right, BorderType.Constant, new MCvScalar(color.B, color.G, color.R));
            resized.Dispose();

            return padded;
        }

        private void PreprocessImage(Mat img, DenseTensor<float> tensor)
        {
            var imageData = img.GetData();
            if (!(imageData is byte[,,] bytes)) return;

            int height = img.Height;
            int width = img.Width;
            int channels = img.NumberOfChannels;
            
            // 使用分块并行处理
            int blockSize = 32; // 每块32行
            int numBlocks = (height + blockSize - 1) / blockSize;
            
            Parallel.For(0, numBlocks, block =>
            {
                int startY = block * blockSize;
                int endY = Math.Min(startY + blockSize, height);
                
                for (int y = startY; y < endY; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        tensor[0, 0, y, x] = bytes[y, x, 2] / 255.0f;  // R
                        tensor[0, 1, y, x] = bytes[y, x, 1] / 255.0f;  // G
                        tensor[0, 2, y, x] = bytes[y, x, 0] / 255.0f;  // B
                    }
                }
            });
        }

        private float[,] ScaleBoxes(float[,] boxes, Size sourceSize, Size targetSize)
        {
            float gain_x = (float)targetSize.Width / sourceSize.Width;
            float gain_y = (float)targetSize.Height / sourceSize.Height;
            float min_gain = Math.Min(gain_x, gain_y);

            float[,] scaled = new float[boxes.GetLength(0), boxes.GetLength(1)];
            for (int i = 0; i < boxes.GetLength(0); i++)
            {
                scaled[i, 0] = boxes[i, 0] * gain_x;  // x1
                scaled[i, 1] = boxes[i, 1] * gain_y;  // y1
                scaled[i, 2] = boxes[i, 2] * gain_x;  // x2
                scaled[i, 3] = boxes[i, 3] * gain_y;  // y2
                scaled[i, 4] = boxes[i, 4];           // conf
                scaled[i, 5] = boxes[i, 5];           // class
            }
            return scaled;
        }

        private ConcurrentDictionary<int, List<Prediction>> ProcessPredictions(float[] result, float proportion_x, float proportion_y, bool agnostic = false)
        {
            var dictionary = new ConcurrentDictionary<int, List<Prediction>>();

            // 使用并行处理预测结果
            Parallel.For(0, _lengthOfPredict, i =>
            {
                int j = i * (_countOfKind + 5);
                float confidence = result[j + 4];
                if (confidence >= this.confidence)
                {
                    int kind = j + 5;
                    for (int k = kind + 1; k < j + _countOfKind + 5; k++)
                    {
                        if (result[k] > result[kind])
                        {
                            kind = k;
                        }
                    }
                    kind = kind % (_countOfKind + 5) - 5;

                    dictionary.AddOrUpdate(agnostic ? 0 : kind,
                        _ => new List<Prediction> { new(kind, result[j], result[j + 1], result[j + 2], result[j + 3], confidence) },
                        (_, list) =>
                        {
                            lock (list)
                            {
                                list.Add(new(kind, result[j], result[j + 1], result[j + 2], result[j + 3], confidence));
                            }
                            return list;
                        });
                }
            });

            // 并行处理NMS
            Parallel.ForEach(dictionary.Keys, key =>
            {
                var predictions = dictionary[key];
                lock (predictions)
                {
                    predictions.Sort();
                    var toRemove = new HashSet<Prediction>();

                    for (int i = 0; i < predictions.Count; i++)
                    {
                        if (toRemove.Contains(predictions[i])) continue;
                        
                        for (int j = i + 1; j < predictions.Count; j++)
                        {
                            if (toRemove.Contains(predictions[j])) continue;
                            
                            if (predictions[i].IOU(predictions[j]) >= iou)
                            {
                                toRemove.Add(predictions[j]);
                            }
                        }
                    }

                    predictions.RemoveAll(p => toRemove.Contains(p));

                    for (int i = 0; i < predictions.Count; i++)
                    {
                        predictions[i].X *= proportion_x;
                        predictions[i].Y *= proportion_y;
                        predictions[i].Width *= proportion_x;
                        predictions[i].Height *= proportion_y;
                    }
                }
            });

            return dictionary;
        }

        /// <summary>
        /// 摆烂进行预测[适用于性能要求不太高的场景]，适当占用CPU
        /// </summary>
        /// <param name="mat">Emgu.CV.Mat图片对象</param>
        /// <returns>预测结果，包含处理后的图像和检测框数据</returns>
        public DetectionResult DetectLetItRot(Mat mat)
        {
            ThrowIfDisposed();
            
            Mat processedImage = null;
            Mat letterboxed = null;
            try
            {
                var originalSize = new Size(mat.Width, mat.Height);
                processedImage = mat.Clone();
                
                // 使用letterbox预处理，保持32的倍数
                letterboxed = LetterboxImage(mat, _modelInputSize, Color.FromArgb(114, 114, 114), stride: 32);
                
                // 确保图像是连续的内存布局
                if (!letterboxed.IsContinuous)
                {
                    var temp = letterboxed.Clone();
                    letterboxed.Dispose();
                    letterboxed = temp;
                }

                lock (_bufferLock)
                {
                    // 预处理图像数据
                    PreprocessImage(letterboxed, _reuseInputTensor);

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_inputMetadataName, _reuseInputTensor)
                    };

                    using var values = inference_session.Run(inputs);
                    float[] result = values.First(value => value.Name == _outputMetadataName).AsEnumerable<float>().ToArray();

                    var predictions = ProcessPredictions(result, 1.0f, 1.0f, true);
                    
                    // 转换预测结果为二维数组并缩放到原始图像尺寸
                    var outputs = ConvertPredictionsToOutput(predictions);
                    outputs = ScaleBoxes(outputs, new Size(letterboxed.Width, letterboxed.Height), originalSize);

                    // Draw boxes on the image
                    for (int i = 0; i < outputs.GetLength(0); i++)
                    {
                        var rect = new Rectangle(
                            (int)outputs[i, 0],
                            (int)outputs[i, 1],
                            (int)(outputs[i, 2] - outputs[i, 0]),
                            (int)(outputs[i, 3] - outputs[i, 1])
                        );
                        
                        CvInvoke.Rectangle(processedImage, rect, new MCvScalar(0, 255, 0), 2);
                        CvInvoke.PutText(processedImage, 
                            $"{(int)outputs[i, 5]} {outputs[i, 4]:F2}", 
                            new Point((int)outputs[i, 0], (int)outputs[i, 1] - 10),
                            FontFace.HersheyTriplex,
                            0.8,
                            new MCvScalar(255, 0, 0),
                            1,
                            LineType.AntiAlias);
                    }

                    return new DetectionResult(processedImage, outputs);
                }
            }
            catch (Exception ex)
            {
                ApplicationLogger.Instance.Error($"Error during detection: {ex.Message}", ex);
                processedImage?.Dispose();
                letterboxed?.Dispose();
                throw;
            }
            finally
            {
                letterboxed?.Dispose();
            }
        }

        private float[,] ConvertPredictionsToOutput(ConcurrentDictionary<int, List<Prediction>> predictions)
        {
            int totalPredictions = predictions.Sum(p => p.Value.Count);
            var outputs = new float[totalPredictions, 6];
            int idx = 0;
            foreach (var kvp in predictions)
            {
                foreach (var pred in kvp.Value)
                {
                    outputs[idx, 0] = pred.X;
                    outputs[idx, 1] = pred.Y;
                    outputs[idx, 2] = pred.X + pred.Width;
                    outputs[idx, 3] = pred.Y + pred.Height;
                    outputs[idx, 4] = pred.Confidence;
                    outputs[idx, 5] = kvp.Key;
                    idx++;
                }
            }
            return outputs;
        }

        /// <summary>
        /// 全力进行预测[适用于性能要求比较高的场景]，超高占用CPU
        /// </summary>
        /// <param name="mat">Emgu.CV.Mat图片对象</param>
        /// <returns>预测结果，包含处理后的图像和检测框数据</returns>
        public DetectionResult DetectAllOut(Mat mat)
        {
            #region yolov5模型一般来说就一个输入和一个输出
            //输入参数的名称
            string InputMetadataName = inference_session!.InputNames[0];
            //输出参数的名称
            string OutputMetadataName = inference_session!.OutputNames[0];
            #endregion

            //获得模型的输入维度
            int[] input_dimensions = inference_session!.InputMetadata[InputMetadataName].Dimensions;

            //计算图片尺寸和模型尺寸的比例
            float proportion_x = 1f * mat.Width / input_dimensions[3];
            float proportion_y = 1f * mat.Height / input_dimensions[2];

            //拷贝一份新的图片对象，避免修改源图像尺寸
            Mat processedImage = mat.Clone();
            using var resizedMat = mat.Clone();
            CvInvoke.Resize(resizedMat, resizedMat, new Size(input_dimensions[3], input_dimensions[2]));

            //根据输入维度创建输入的稠密张量
            var dense_tensor = new DenseTensor<float>(input_dimensions);
            var image = (byte[,,])resizedMat.GetData();

            //使用Parallel.For加速预处理
            Parallel.For(0, resizedMat.Height, i =>
            {
                for (int j = 0; j < resizedMat.Width; j++)
                {
                    dense_tensor[0, 0, i, j] = image[i, j, 0] / 255f;
                    dense_tensor[0, 1, i, j] = image[i, j, 1] / 255f;
                    dense_tensor[0, 2, i, j] = image[i, j, 2] / 255f;
                }
            });

            //构建输入并运行推理
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(InputMetadataName, dense_tensor) };
            using var results = inference_session.Run(inputs);
            var resultsArray = results.First().AsEnumerable<float>().ToArray();
            var predictions = ProcessResults(resultsArray, proportion_x, proportion_y);

            // Draw boxes on the image
            foreach (var kvp in predictions)
            {
                foreach (var pred in kvp.Value)
                {
                    var rect = new Rectangle(
                        (int)pred.X,
                        (int)pred.Y,
                        (int)pred.Width,
                        (int)pred.Height
                    );
                    CvInvoke.Rectangle(processedImage, rect, new MCvScalar(0, 255, 0), 2);
                    CvInvoke.PutText(processedImage, 
                        $"{kvp.Key} {pred.Confidence:F2}", 
                        new Point((int)pred.X, (int)pred.Y - 10),
                        FontFace.HersheyTriplex,
                        0.8,
                        new MCvScalar(255, 0, 0),
                        1,
                        LineType.AntiAlias);
                }
            }

            // Convert predictions to output format similar to Python
            int totalPredictions = predictions.Sum(p => p.Value.Count);
            var outputs = new float[totalPredictions, 6];
            int idx = 0;
            foreach (var kvp in predictions)
            {
                foreach (var pred in kvp.Value)
                {
                    outputs[idx, 0] = pred.X;
                    outputs[idx, 1] = pred.Y;
                    outputs[idx, 2] = pred.X + pred.Width;
                    outputs[idx, 3] = pred.Y + pred.Height;
                    outputs[idx, 4] = pred.Confidence;
                    outputs[idx, 5] = kvp.Key;
                    idx++;
                }
            }

            return new DetectionResult(processedImage, outputs);
        }

        private ConcurrentDictionary<int, List<Prediction>> ProcessResults(float[] resultsArray, float proportionX, float proportionY)
        {
            var dictionary = new ConcurrentDictionary<int, List<Prediction>>();
            var outputDimensions = inference_session!.OutputMetadata[inference_session!.OutputNames[0]].Dimensions;

            var lengthOfPredict = outputDimensions[1];
            var countOfKind = outputDimensions[2] - 5;

            // 解析输出并过滤掉低置信度的预测结果
            Parallel.For(0, lengthOfPredict, i =>
            {
                int baseIndex = i * (countOfKind + 5);
                float confidence = resultsArray[baseIndex + 4];
                
                if (confidence >= this.confidence)
                {
                    // 找到最大置信度的类别
                    float maxClassScore = float.MinValue;
                    int predictedClass = 0;
                    
                    for (int j = 0; j < countOfKind; j++)
                    {
                        float classScore = resultsArray[baseIndex + 5 + j];
                        if (classScore > maxClassScore)
                        {
                            maxClassScore = classScore;
                            predictedClass = j;
                        }
                    }

                    // 创建预测对象
                    var prediction = new Prediction(
                        predictedClass,
                        resultsArray[baseIndex],     // x
                        resultsArray[baseIndex + 1], // y
                        resultsArray[baseIndex + 2], // width
                        resultsArray[baseIndex + 3], // height
                        confidence * maxClassScore    // 最终置信度是目标置信度和类别置信度的乘积
                    );

                    dictionary.AddOrUpdate(
                        predictedClass,
                        _ => new List<Prediction> { prediction },
                        (_, list) =>
                        {
                            lock (list)
                            {
                                list.Add(prediction);
                            }
                            return list;
                        }
                    );
                }
            });

            // NMS处理
            var kinds = dictionary.Keys.ToList();
            Parallel.ForEach(kinds, kind =>
            {
                var predictions = dictionary[kind];
                predictions.Sort();

                var toRemove = new HashSet<Prediction>();
                for (int i = 0; i < predictions.Count; i++)
                {
                    if (toRemove.Contains(predictions[i])) continue;
                    
                    for (int j = i + 1; j < predictions.Count; j++)
                    {
                        if (toRemove.Contains(predictions[j])) continue;
                        
                        if (predictions[i].IOU(predictions[j]) >= iou)
                        {
                            toRemove.Add(predictions[j]);
                        }
                    }
                }

                predictions.RemoveAll(p => toRemove.Contains(p));

                // 缩放回原始图像尺寸
                foreach (var pred in predictions)
                {
                    pred.X *= proportionX;
                    pred.Y *= proportionY;
                    pred.Width *= proportionX;
                    pred.Height *= proportionY;
                }
            });

            return dictionary;
        }

        public class DetectionResult
        {
            public Mat ProcessedImage { get; set; }
            public float[,] Outputs { get; set; }

            public DetectionResult(Mat processedImage, float[,] outputs)
            {
                ProcessedImage = processedImage;
                Outputs = outputs;
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
                    _resizedMat?.Dispose();
                    _reuseInputTensor = null;
                    _imageBuffer = null;
                    _processingBuffer = null;
                    
                    // Don't dispose inference_session here as it's managed by ModelManager
                    inference_session = null;
                    
                    // Clear any other managed resources
                    yolo_onnx_path = null;
                }
                _disposed = true;
            }
        }

        ~Yolov5Onnx()
        {
            Dispose(false);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Yolov5Onnx));
            }
        }
    }
}
