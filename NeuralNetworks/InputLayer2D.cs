using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class InputLayer2D : Layer2D
    {
        public InputLayer2D(int nMaps, int width, int height)
        {
            _nMaps = nMaps;
            _nRows = height;
            _nCols = width;
            _maps = new NeuronMap[nMaps];
            
            for (int m = 0; m < _nMaps; m++)
            {
                _maps[m] = new NeuronMap(height, width);
                for (int r = 0; r < _nRows; r++)
                {
                    for (int c = 0; c < _nCols; c++)
                    {
                        _maps[m].Neurons[r, c] = new Neuron();
                    }
                }
            }
        }

        public InputLayer2D(StreamReader reader)
        {
            string[] line = reader.ReadLine().Split();
            _nMaps = int.Parse(line[0]);
            _nRows = int.Parse(line[1]);
            _nCols = int.Parse(line[2]);

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

        internal override void BindInputFrom(Layer prevLayer, bool randomizeWeights)
        {
            throw new NotImplementedException();
        }

        internal override float[] Propagate(bool realRun)
        {
            throw new NotImplementedException();
        }

        internal override float PropagateAndLearn(float[] targetOutput, float learningRate)
        {
            throw new NotImplementedException();
        }

        internal override void BackPropagate(float learningRate)
        {
            return;
        }

        internal override void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(typeof(InputLayer2D));
            writer.WriteLine("{0} {1} {2}", _nMaps, _nRows, _nCols);
        }
        
        internal float[] RunNetwork(float[][,] input, bool realRun)
        {
            for (int m = 0; m < _nMaps; m++)
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].Value = input[m][r, c];
            return _nextLayer.Propagate(realRun);
        }

        internal float Train(float[][,] input, float[] targetOutput, float learningRate)
        {
            for (int m = 0; m < _nMaps; m++)
                for (int r = 0; r < _nRows; r++)
                    for (int c = 0; c < _nCols; c++)
                        _maps[m].Neurons[r, c].Value = input[m][r, c];
            return _nextLayer.PropagateAndLearn(targetOutput, learningRate);
        }
    }
}
