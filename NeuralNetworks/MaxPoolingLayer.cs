using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class MaxPoolingLayer : PoolingLayer
    {
        private NeuronMap[] _prevMaps;

        public MaxPoolingLayer(int poolWidth, int poolHeight)
            : base(poolWidth, poolHeight)
        {

        }

        public MaxPoolingLayer(StreamReader reader)
        {
            string[] line = reader.ReadLine().Split();
            _poolRows = int.Parse(line[0]);
            _poolCols = int.Parse(line[1]);
        }

        internal override void BindInputFrom(Layer prevLayer, bool randomizeWeights)
        {
            _prevLayer = prevLayer;

            Layer2D prevLayer2D = (Layer2D)prevLayer;
            _nMaps = prevLayer2D.N_Maps;
            _nRows = prevLayer2D.N_Rows / _poolRows;
            _nCols = prevLayer2D.N_Cols / _poolCols;
            _prevMaps = prevLayer2D.Maps;
            _maps = new NeuronMap[_nMaps];
            for (int m = 0; m < _nMaps; m++)
            {
                _maps[m] = new NeuronMap(_nRows, _nCols);
                for (int r = 0; r < _nRows; r++)
                {
                    for (int c = 0; c < _nCols; c++)
                    {
                        _maps[m].Neurons[r, c] = new Neuron();
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
            Parallel.For(0, _nMaps, m =>
            {
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].BackProp();
            });
            _prevLayer.BackPropagate(learningRate);
        }

        internal override void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(typeof(MaxPoolingLayer));
            writer.WriteLine("{0} {1}", _poolRows, _poolCols);
        }

        private void UpdateNeuronValues()
        {
            Parallel.For(0, _nMaps, m =>
            {
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].Imitate(_prevMaps[m].GetMaxInSection(r * _poolRows, c * _poolCols, _poolRows, _poolCols));
            });
        }
    }
}
