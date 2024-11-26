using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BingLing.Yolov5Onnx.Gpu
{
    /// <summary>
    /// 预测结果，实现IComparable接口，使得其内部具备置信度从大到小排序的能力
    /// </summary>
    public class Prediction(int kind, float x, float y, float width, float height, float confidense) : IComparable<Prediction>
    {

        /// <summary>
        /// 预测结果种类
        /// </summary>
        public int Kind { set; get; } = kind;

        /// <summary>
        /// 预测结果中心横坐标
        /// </summary>
        public float X { set; get; } = x;

        /// <summary>
        /// 预测结果中心纵坐标
        /// </summary>
        public float Y { set; get; } = y;

        /// <summary>
        /// 预测结果宽度
        /// </summary>
        public float Width { set; get; } = width;

        /// <summary>
        /// 预测结果高度
        /// </summary>
        public float Height { set; get; } = height;

        /// <summary>
        /// 预测结果置信度
        /// </summary>
        public float Confidence { set; get; } = confidense;

        /// <summary>
        /// 避免频繁乘法运算而设的属性
        /// </summary>
        public float Area { set; get; } = width * height;

        /// <summary>
        /// 比较规则
        /// </summary>
        /// <param name="other">另一个预测结果</param>
        /// <returns></returns>
        public int CompareTo(Prediction? other)
        {
            if (Confidence > other!.Confidence)
            {
                return -1;
            }
            return 1;
        }

        /// <summary>
        /// 与另一个预测结果的交并比
        /// </summary>
        /// <param name="other">另一个预测结果</param>
        /// <returns></returns>
        public float IOU(Prediction other)
        {
            float minX = Math.Max(X - Width / 2, other.X - other.Width / 2);
            float maxX = Math.Min(X + Width / 2, other.X + other.Width / 2);
            if (maxX < minX)
            {
                return 0;
            }

            float minY = Math.Max(Y - Height / 2, other.Y - other.Height / 2);
            float maxY = Math.Min(Y + Height / 2, other.Y + other.Height / 2);
            if (maxY < minY)
            {
                return 0;
            }

            float intersection = (maxX - minX) * (maxY - minY);
            float union = Area + other.Area - intersection;
            return intersection / union;
        }
    }
}
