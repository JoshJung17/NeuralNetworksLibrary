using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNetworks
{
    public class NeuralNetwork1D
    {
        private InputLayer _head;
        private List<Layer> _layers;
        internal static Random Rand = new Random();
        private int _iterations;

        public NeuralNetwork1D(InputLayer head)
        {
            _head = head;
            _layers = new List<Layer>();
            _layers.Add(head);
        }

        /// <summary>
        /// Initializes an instance of a neural network given a text stream
        /// </summary>
        /// <param name="savePath">The input stream to read from</param>
        public NeuralNetwork1D(string savePath)
        {
            using (StreamReader reader = new StreamReader(savePath))
            {
                _iterations = int.Parse(reader.ReadLine());
                int nLayers = int.Parse(reader.ReadLine());
                _head = (InputLayer)Layer.GetLayerFrom(reader);
                _layers = new List<Layer>();
                _layers.Add(_head);
                for (int l = 1; l < nLayers; l++)
                {
                    Layer layer = Layer.GetLayerFrom(reader);
                    if (_layers.Last() is Layer2D) ((Layer2D)_layers.Last()).BindTo(layer, false);
                    else ((Layer1D)_layers.Last()).BindTo((Layer1D)layer, false);
                    _layers.Add(layer);
                }
            }
        }

        /// <summary>
        /// Saves the entire network to a file, including the structure and parameters
        /// </summary>
        /// <param name="savePath">The location of the file to save to</param>
        public void SaveNetwork(string savePath)
        {
            using (StreamWriter writer = new StreamWriter(savePath))
            {
                writer.WriteLine(_iterations);
                writer.WriteLine(_layers.Count);
                for (int l = 0; l < _layers.Count; l++)
                {
                    _layers[l].SaveWeights(writer);
                }
            }
        }

        /// <summary>
        /// Computes the output of the neural network given an array of input
        /// </summary>
        /// <param name="input">The input to use in the network</param>
        /// <param name="realRun">True for testing, false for training</param>
        /// <returns>An array representing the output of the network</returns>
        public float[] RunNetwork(float[] input, bool realRun)
        {
            return _head.RunNetwork(input, realRun);
        }

        public float RunBackProp(float[] targetOutput, float initialLearningRate, float initialHalfRate)
        {
            _iterations++;
            float learningRate = initialLearningRate * initialHalfRate / (_iterations + initialHalfRate);
            return ((OutputLayer)_layers.Last()).RunBackProp(targetOutput, learningRate);
        }

        public float Train(float[] input, float[] targetOutput, float initialLearningRate, float initialHalfRate)
        {
            _iterations++;
            float learningRate = initialLearningRate * initialHalfRate / (_iterations + initialHalfRate);
            return _head.Train(input, targetOutput, learningRate);
        }

        public void AddLayer(Layer layer)
        {
            if (_layers.Last() is Layer2D) ((Layer2D)_layers.Last()).BindTo(layer, true);
            else ((Layer1D)_layers.Last()).BindTo((Layer1D)layer, true);
            _layers.Add(layer);
        }

        public int Iterations
        {
            get
            {
                return _iterations;
            }
        }
    }
}
