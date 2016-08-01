using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    /// <summary>
    /// Represents a convolution layer composed of one or more feature maps
    /// </summary>
    public class ConvolutionLayer : Layer2D
    {
        private int _kernelRows, _kernelCols;
        private WeightSet[] _weightSets;
        private int _activationFunction;

        private float _weightUpdateAdjustment;

        /// <summary>
        /// Initializes an instance of a convolution layer
        /// </summary>
        /// <param name="nMaps">The number of feature maps</param>
        /// <param name="kernelWidth">The width of the kernel to be used for convolution</param>
        /// <param name="kernelHeight">The height of the kernel to be used for convolution</param>
        /// <param name="activationFunction">The activation function constant, defined in NeuralNetwork2D</param>
        public ConvolutionLayer(int nMaps, int kernelWidth, int kernelHeight, int activationFunction)
        {
            _nMaps = nMaps;
            _maps = new NeuronMap[nMaps];

            _kernelRows = kernelHeight;
            _kernelCols = kernelWidth;
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// Initializes an instance of a convolution layer given a text stream
        /// </summary>
        /// <param name="reader">The input stream to read from</param>
        public ConvolutionLayer(StreamReader reader)
        {
            string[] line = reader.ReadLine().Split();
            _nMaps = int.Parse(line[0]);
            _maps = new NeuronMap[_nMaps];
            _kernelRows = int.Parse(line[1]);
            _kernelCols = int.Parse(line[2]);
            _activationFunction = int.Parse(line[3]);
            _weightSets = new WeightSet[_nMaps];
            for (int m = 0; m < _nMaps; m++)
                _weightSets[m] = WeightSet.GetWeightSetFrom(reader);
        }
        
        internal override void BindInputFrom(Layer prevLayer, bool randomizeWeights)
        {
            _prevLayer = prevLayer;

            if (randomizeWeights)
                _weightSets = new WeightSet[_nMaps];

            Layer2D prevLayer2D = (Layer2D)prevLayer;
            _nRows = prevLayer2D.N_Rows - _kernelRows + 1;
            _nCols = prevLayer2D.N_Cols - _kernelCols + 1;
            _weightUpdateAdjustment = (float)(1.0 / Math.Sqrt(_nRows * _nCols));
            int nPrevMaps = prevLayer2D.N_Maps;
            for (int m = 0; m < _nMaps; m++)
            {
                if (randomizeWeights)
                {
                    WeightSet w = new WeightSet(nPrevMaps * _kernelRows * _kernelCols, NeuralNetwork2D.Rand);
                    _weightSets[m] = w;
                }
                _maps[m] = new NeuronMap(_nRows, _nCols,_weightSets[m]);
                Neuron[,] neurons = _maps[m].Neurons;
                for (int r = 0; r < _nRows; r++)
                {
                    for (int c = 0; c < _nCols; c++)
                    {
                        Neuron[] inputNeurons = new Neuron[nPrevMaps * _kernelRows * _kernelCols];
                        for (int pm = 0; pm < nPrevMaps; pm++)
                        {
                            Neuron[] inputFromMap = prevLayer2D.Maps[pm].GetSubSection(r, c, _kernelRows, _kernelCols);
                            Array.Copy(inputFromMap, 0, inputNeurons, pm * _kernelRows * _kernelCols, _kernelRows * _kernelCols);
                        }
                        neurons[r, c] = new Neuron(inputNeurons, _weightSets[m], _activationFunction);
                    }
                }
            }
        }

        internal override float[] Propagate(bool realRun)
        {
            UpdateNeuronValues();
            return _nextLayer.Propagate(realRun);
        }

        internal override float PropagateAndLearn(float[] targetOutput, float learningRate)
        {
            UpdateNeuronValues();
            return _nextLayer.PropagateAndLearn(targetOutput, learningRate);
        }

        internal override void BackPropagate(float learningRate)
        {
            float lr = learningRate * _weightUpdateAdjustment;
            Parallel.For(0, _nMaps, m =>
            {
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].BackProp();
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].UpdateWeights(lr);
            });
            _prevLayer.BackPropagate(learningRate);
        }

        internal override void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(typeof(ConvolutionLayer));
            writer.WriteLine("{0} {1} {2} {3}", _nMaps, _kernelRows, _kernelCols, _activationFunction);
            for (int m = 0; m < _nMaps; m++)
                _maps[m].SaveWeights(writer);
        }

        /// <summary>
        /// Updates all neuron values with an activation function applied to the sum of products of weights and outputs from the previous layer
        /// </summary>
        private void UpdateNeuronValues()
        {
            Parallel.For(0, _nMaps, m =>
            {
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].UpdateValue();
            });
        }
    }
}
