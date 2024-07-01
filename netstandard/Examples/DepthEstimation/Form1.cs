using DepthONNX;
using UMapx.Core;
using UMapx.Imaging;

namespace DepthEstimation
{
    public partial class Form1 : Form
    {
        private readonly DepthEstimator _depthEstimator;

        public Form1()
        {
            InitializeComponent();

            BackgroundImageLayout = ImageLayout.Zoom;
            DragDrop += Form1_DragDrop;
            DragEnter += Form1_DragEnter;
            AllowDrop = true;
            Text = "DepthONNX: Depth estimation";

            _depthEstimator = new DepthEstimator(DepthEstimatorQuality.High);
            var image = new Bitmap("example.png");
            Process(image);
        }

        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.All : DragDropEffects.None;
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            Cursor = Cursors.WaitCursor;
            var file = ((string[])e.Data.GetData(DataFormats.FileDrop, true))[0];
            var image = new Bitmap(file);
            Process(image);
            Cursor = Cursors.Default;
        }

        private void Process(Bitmap image)
        {
            var results = _depthEstimator.Forward(
                image: image,
                interpolationMode: InterpolationMode.Bicubic);

            var mask = results.FromGrayscale();
            image?.Dispose();

            //mask.Save("output.png", System.Drawing.Imaging.ImageFormat.Png);

            BackgroundImage?.Dispose();
            BackgroundImage = mask;
        }

    }
}