using Emgu.CV;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Drawing;

namespace BingLing.Yolov5Onnx.Gpu
{
    /// <summary>
    /// Yolo类
    /// </summary>
    public class Yolov5Onnx
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

        private readonly InferenceSession? inference_session;

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
        }

        /// <summary>
        /// 摆烂进行预测[适用于性能要求不太高的场景]，适当占用CPU
        /// </summary>
        /// <param name="mat">Emgu.CV.Mat图片对象</param>
        /// <returns>预测结果字典，key为类型，value为预测结果</returns>
        public ConcurrentDictionary<int, List<Prediction>> DetectLetItRot(Mat mat)
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
            mat = mat.Clone();

            //缩放至模型输入大小
            CvInvoke.Resize(mat, mat, new Size(input_dimensions[3], input_dimensions[2]));

            //根据输入维度创建输入的稠密张量
            DenseTensor<float> dense_tensor = new(input_dimensions);

            //yolov5预测的一般都是彩色图[三通道]，即时传入图片的是灰度图也无所谓，只要模型是预测三通道图片的就可以
            byte[,,] image = (byte[,,])mat.GetData();
            for (int i = 0; i < mat.Height; i++)
            {
                for (int j = 0; j < mat.Width; j++)
                {
                    dense_tensor[0, 0, i, j] = (float)(image[i, j, 0] / 255f);
                    dense_tensor[0, 1, i, j] = (float)(image[i, j, 1] / 255f);
                    dense_tensor[0, 2, i, j] = (float)(image[i, j, 2] / 255f);
                }
            }

            //构建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(InputMetadataName, dense_tensor)
            };

            //推理预测
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> values = inference_session.Run(inputs);

            //把推理结果转换成一维数组
            float[] result = values.First(value => value.Name == OutputMetadataName).AsEnumerable<float>().ToArray();
            //释放资源
            values.Dispose();

            ConcurrentDictionary<int, List<Prediction>> dictionary = new();
            int[] output_dimensions = inference_session.OutputMetadata[OutputMetadataName].Dimensions;

            int length_of_predict = output_dimensions[1];
            int count_of_kind = output_dimensions[2] - 5;
            //解析输出并过滤掉低置信度的预测结果
            for (int i = 0; i < length_of_predict; i++)
            {
                int j = i * (count_of_kind + 5);
                float confidence = result[j + 4];
                if (confidence >= this.confidence)
                {
                    int kind = j + 5;
                    for (int k = kind + 1; k < j + count_of_kind + 5; k++)
                    {
                        if (result[k] > result[kind])
                        {
                            kind = k;
                        }
                    }
                    kind = kind % (count_of_kind + 5) - 5;

                    if (!dictionary.ContainsKey(kind))
                    {
                        dictionary.TryAdd(kind, new List<Prediction>());
                    }

                    dictionary[kind].Add(new Prediction(kind, result[j], result[j + 1], result[j + 2], result[j + 3], confidence));
                }
            }

            //NMS算法[同一种预测类别且交并比大于设定阈值的两个预测结果视为同一个目标]，去除针对同一目标的多余预测结果
            List<int> kinds = new List<int>(dictionary.Keys);
            foreach (var kind in kinds)
            {
                List<Prediction> predictions = dictionary[kind];
                predictions.Sort();

                HashSet<Prediction> hashSet = new HashSet<Prediction>();
                for (int i = 0; i < predictions.Count; i++)
                {
                    if (hashSet.Contains(predictions[i]))
                    {
                        continue;
                    }
                    for (int j = i + 1; j < predictions.Count; j++)
                    {
                        if (hashSet.Contains(predictions[j]))
                        {
                            continue;
                        }

                        if (predictions[i].IOU(predictions[j]) >= iou)
                        {
                            hashSet.Add(predictions[j]);
                        }
                    }
                }

                foreach (var item in hashSet)
                {
                    predictions.Remove(item);
                }

                //根据比例缩放回来
                foreach (var prediction in predictions)
                {
                    prediction.X *= proportion_x;
                    prediction.Y *= proportion_y;
                    prediction.Width *= proportion_x;
                    prediction.Height *= proportion_y;
                }
            }

            return dictionary;
        }

        /// <summary>
        /// 全力进行预测[适用于性能要求比较高的场景]，超高占用CPU
        /// </summary>
        /// <param name="mat">Emgu.CV.Mat图片对象</param>
        /// <returns>预测结果字典，key为类型，value为预测结果</returns>
        public ConcurrentDictionary<int, List<Prediction>> DetectAllOut(Mat mat)
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
            return ProcessResults(resultsArray, proportion_x, proportion_y);
        }

        private ConcurrentDictionary<int, List<Prediction>> ProcessResults(float[] resultsArray, float proportion_x, float proportion_y)
        {
            var dictionary = new ConcurrentDictionary<int, List<Prediction>>();
            int[] output_dimensions = inference_session!.OutputMetadata[inference_session!.OutputNames[0]].Dimensions;

            int length_of_predict = output_dimensions[1];
            int count_of_kind = output_dimensions[2] - 5;
            //解析输出并过滤掉低置信度的预测结果
            Parallel.For(0, length_of_predict, i =>
            {
                int j = i * (count_of_kind + 5);
                float confidence = resultsArray[j + 4];
                if (confidence >= this.confidence)
                {
                    int kind = j + 5;
                    for (int k = kind + 1; k < j + count_of_kind + 5; k++)
                    {
                        if (resultsArray[k] > resultsArray[kind])
                        {
                            kind = k;
                        }
                    }
                    kind = kind % (count_of_kind + 5) - 5;

                    if (!dictionary.ContainsKey(kind))
                    {
                        dictionary.TryAdd(kind, new List<Prediction>());
                    }
                    lock (dictionary[kind])
                    {
                        dictionary[kind].Add(new Prediction(kind, resultsArray[j], resultsArray[j + 1], resultsArray[j + 2], resultsArray[j + 3], confidence));
                    }
                }
            });

            //NMS算法[同一种预测类别且交并比大于设定阈值的两个预测结果视为同一个目标]，去除针对同一目标的多余预测结果
            List<int> kinds = new List<int>(dictionary.Keys);
            Parallel.ForEach(kinds, kind =>
            {
                List<Prediction> predictions = dictionary[kind];
                predictions.Sort();

                HashSet<Prediction> hashSet = new HashSet<Prediction>();
                for (int i = 0; i < predictions.Count; i++)
                {
                    if (hashSet.Contains(predictions[i]))
                    {
                        continue;
                    }
                    for (int j = i + 1; j < predictions.Count; j++)
                    {
                        if (hashSet.Contains(predictions[j]))
                        {
                            continue;
                        }

                        if (predictions[i].IOU(predictions[j]) >= iou)
                        {
                            hashSet.Add(predictions[j]);
                        }
                    }
                }

                foreach (var item in hashSet)
                {
                    predictions.Remove(item);
                }

                //根据比例缩放回来
                Parallel.ForEach(predictions, prediction =>
                {
                    prediction.X *= proportion_x;
                    prediction.Y *= proportion_y;
                    prediction.Width *= proportion_x;
                    prediction.Height *= proportion_y;
                });
            });

            return dictionary;
        }

        /// <summary>
        /// 析构函数，释放yolov5对象时，把推理会话释放掉避免内存泄漏
        /// </summary>
        ~Yolov5Onnx()
        {
            this.inference_session?.Dispose();
        }
    }
}
