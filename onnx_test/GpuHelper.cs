using System.Runtime.InteropServices;
using System.Text;
using onnx_test;

namespace BingLing.Yolov5Onnx.Gpu
{
    public static class GpuHelper
    {
        [DllImport("nvml.dll", CharSet = CharSet.Ansi)]
        private static extern int nvmlInit_v2();

        [DllImport("nvml.dll", CharSet = CharSet.Ansi)]
        private static extern int nvmlDeviceGetCount_v2(out int deviceCount);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi)]
        private static extern int nvmlDeviceGetHandleByIndex_v2(int index, out IntPtr device);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi)]
        private static extern int nvmlDeviceGetName(IntPtr device, [MarshalAs(UnmanagedType.LPStr)] StringBuilder name, int length);

        public static List<string> GetAvailableGpus()
        {
            var gpus = new List<string>();
            try
            {
                if (nvmlInit_v2() == 0)
                {
                    int deviceCount;
                    if (nvmlDeviceGetCount_v2(out deviceCount) == 0)
                    {
                        for (int i = 0; i < deviceCount; i++)
                        {
                            IntPtr device;
                            if (nvmlDeviceGetHandleByIndex_v2(i, out device) == 0)
                            {
                                var name = new StringBuilder(64);
                                if (nvmlDeviceGetName(device, name, name.Capacity) == 0)
                                {
                                    gpus.Add($"GPU {i}: {name}");
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                ApplicationLogger.Instance.Error($"Error getting GPU info: {ex.Message}", ex);
            }
            return gpus;
        }
    }
}
