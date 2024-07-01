using System;
using System.Drawing;
using UMapx.Core;

namespace DepthONNX
{
    /// <summary>
    /// Defines depth estimator interface.
    /// </summary>
    public interface IDepthEstimator : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns depth estimation results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="interpolationMode">Interpolation mode</param>
        /// <returns>Result</returns>
        float[,] Forward(Bitmap image, InterpolationMode interpolationMode = InterpolationMode.Bilinear);

        /// <summary>
        /// Returns depth estimation results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="interpolationMode">Interpolation mode</param>
        /// <returns>Result</returns>
        float[,] Forward(float[][,] image, InterpolationMode interpolationMode = InterpolationMode.Bilinear);

        #endregion
    }
}
