using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class NeuronMap
    {
        internal Neuron[,] Neurons;
        private int _nRows, _nCols;
        internal WeightSet Weights;

        internal NeuronMap(int nRows, int nCols)
        {
            _nRows = nRows;
            _nCols = nCols;
            Neurons = new Neuron[nRows, nCols];
        }

        internal NeuronMap(int nRows, int nCols, WeightSet weights)
        {
            _nRows = nRows;
            _nCols = nCols;
            Neurons = new Neuron[nRows, nCols];
            Weights = weights;
        }

        internal Neuron[] GetSubSection(int row, int col, int nRows, int nCols)
        {
            Neuron[] ret = new Neuron[nRows * nCols];
            for (int r = row; r < row+nRows; r++)
                for (int c = col; c < col+nCols; c++)
                    ret[(r - row) * nCols + (c-col)] = Neurons[r, c];
            return ret;
        }

        internal Neuron[] GetAllNeurons()
        {
            Neuron[] ret = new Neuron[_nRows * _nCols];
            for (int r = 0; r < _nRows; r++)
                for (int c = 0; c < _nCols; c++)
                    ret[r * _nCols + c] = Neurons[r, c];
            return ret;
        }

        internal Neuron GetMaxInSection(int row, int col, int nRows, int nCols)
        {
            Neuron ret = Neurons[0,0];
            float maxi = float.NegativeInfinity;
            for (int r = row; r < row+nRows; r++)
                for (int c = col; c < col+nCols; c++)
                    if (Neurons[r,c].Value>maxi)
                    { 
                        maxi = Neurons[r, c].Value;
                        ret = Neurons[r, c];
                    }
            return ret;
        }

        internal void SaveWeights(StreamWriter writer)
        {
            Weights.SaveWeights(writer);
        }
    }
}
