using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Cuda;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

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

        /// <summary>
        /// 摆烂进行预测[适用于性能要求不太高的场景]，适当占用CPU
        /// </summary>
        /// <param name="mat">Emgu.CV.Mat图片对象</param>
        /// <returns>预测结果，包含处理后的图像和检测框数据</returns>
        public DetectionResult DetectLetItRot(Mat mat)
        {
            ThrowIfDisposed();
            
            float proportion_x = 1f * mat.Width / _modelInputSize.Width;
            float proportion_y = 1f * mat.Height / _modelInputSize.Height;

            Mat processedImage = null;
            try
            {
                processedImage = mat.Clone();
                
                // 使用LetterboxImage进行预处理
                using var letterboxed = LetterboxImage(mat, _modelInputSize, new MCvScalar(114, 114, 114));
                
                lock (_bufferLock)
                {
                    Marshal.Copy(letterboxed.DataPointer, _imageBuffer, 0, _imageBuffer.Length);

                    // 使用SIMD优化的并行处理
                    int pixelCount = _modelInputSize.Width * _modelInputSize.Height;
                    int vectorSize = 4;
                    int remainingStart = (pixelCount / vectorSize) * vectorSize;

                    Parallel.For(0, pixelCount / vectorSize, i =>
                    {
                        int baseIdx = i * vectorSize * 3;
                        for (int j = 0; j < vectorSize; j++)
                        {
                            int pixelIdx = i * vectorSize + j;
                            int row = pixelIdx / _modelInputSize.Width;
                            int col = pixelIdx % _modelInputSize.Width;
                            int srcIdx = baseIdx + j * 3;

                            // BGR to RGB conversion
                            _reuseInputTensor[0, 2, row, col] = _imageBuffer[srcIdx] / 255f;     // B -> R
                            _reuseInputTensor[0, 1, row, col] = _imageBuffer[srcIdx + 1] / 255f; // G -> G
                            _reuseInputTensor[0, 0, row, col] = _imageBuffer[srcIdx + 2] / 255f; // R -> B
                        }
                    });

                    // 处理剩余的像素
                    for (int i = remainingStart; i < pixelCount; i++)
                    {
                        int row = i / _modelInputSize.Width;
                        int col = i % _modelInputSize.Width;
                        int srcIdx = i * 3;

                        // BGR to RGB conversion
                        _reuseInputTensor[0, 2, row, col] = _imageBuffer[srcIdx] / 255f;     // B -> R
                        _reuseInputTensor[0, 1, row, col] = _imageBuffer[srcIdx + 1] / 255f; // G -> G
                        _reuseInputTensor[0, 0, row, col] = _imageBuffer[srcIdx + 2] / 255f; // R -> B
                    }

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_inputMetadataName, _reuseInputTensor)
                    };

                    using var values = inference_session.Run(inputs);
                    float[] result = values.First(value => value.Name == _outputMetadataName).AsEnumerable<float>().ToArray();

                    var predictions = ProcessPredictions(result, proportion_x, proportion_y);
                    
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

                    var outputs = ConvertPredictionsToOutput(predictions);
                    return new DetectionResult(processedImage, outputs);
                }
            }
            catch (Exception)
            {
                processedImage?.Dispose();
                throw;
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

        private ConcurrentDictionary<int, List<Prediction>> ProcessPredictions(float[] result, float proportion_x, float proportion_y)
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

                    dictionary.AddOrUpdate(kind,
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
            Parallel.ForEach(dictionary.Keys, kind =>
            {
                var predictions = dictionary[kind];
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

                    // 批量移除
                    predictions.RemoveAll(p => toRemove.Contains(p));

                    // 批量缩放
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

        /// <summary>
        /// 优化的图像缩放方法
        /// </summary>
        private void OptimizedResize(Mat src, Mat dst, Size size, Inter interpolation = Inter.Linear)
        {
            if (!CudaInvoke.HasCuda)
            {
                CvInvoke.Resize(src, dst, size, 0, 0, interpolation);
                return;
            }

            using (GpuMat gpuMatSrc = new GpuMat())
            using (GpuMat gpuMatDst = new GpuMat())
            {
                gpuMatSrc.Upload(src);
                CudaInvoke.Resize(gpuMatSrc, gpuMatDst, size);
                gpuMatDst.Download(dst);
            }
        }

        /// <summary>
        /// 使用letterbox方法调整图像大小，保持宽高比
        /// </summary>
        private Mat LetterboxImage(Mat img, Size newShape, MCvScalar color, bool auto = true, bool scaleFill = false, bool scaleup = true, int stride = 32)
        {
            // 计算缩放比例和填充
            float ratio = Math.Min((float)newShape.Width / img.Width, (float)newShape.Height / img.Height);
            if (!scaleup)
            {
                ratio = Math.Min(ratio, 1.0f);
            }

            // 计算新的未填充尺寸
            int newUnpadWidth = (int)Math.Round(img.Width * ratio);
            int newUnpadHeight = (int)Math.Round(img.Height * ratio);

            int dw = newShape.Width - newUnpadWidth;
            int dh = newShape.Height - newUnpadHeight;

            if (auto) // 最小矩形
            {
                dw %= stride;
                dh %= stride;
            }

            // 均匀分配填充
            int dw_2 = dw / 2;
            int dh_2 = dh / 2;

            // 创建输出图像
            Mat resized = new Mat();
            
            // 选择合适的插值方法
            Inter interpolation = ratio > 1 ? 
                Inter.Area :  // 缩小时使用Area插值
                Inter.Linear;  // 放大时使用LinearExact插值
            
            // 使用优化的缩放方法
            OptimizedResize(img, resized, new Size(newUnpadWidth, newUnpadHeight), interpolation);

            Mat result = new Mat(newShape, img.Depth, img.NumberOfChannels);
            result.SetTo(color);

            // 复制调整大小的图像到填充图像的中心
            if (dw_2 > 0 && dh_2 > 0)
            {
                var roi = new Rectangle(dw_2, dh_2, newUnpadWidth, newUnpadHeight);
                Mat resultRoi = new Mat(result, roi);
                resized.CopyTo(resultRoi);
            }
            else
            {
                resized.CopyTo(result);
            }

            resized.Dispose();
            return result;
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
