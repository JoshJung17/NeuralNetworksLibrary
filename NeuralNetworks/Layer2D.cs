using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public abstract class Layer2D : Layer
    {
        protected NeuronMap[] _maps;
        protected int _nMaps, _nRows, _nCols;

        protected Layer2D()
        {

        }

        internal override Neuron[] GetAllNeurons()
        {
            Neuron[] ret = new Neuron[_nMaps * _nRows * _nCols];
            for (int map = 0; map < _nMaps; map++)
            {
                Array.Copy(_maps[map].GetAllNeurons(), 0, ret, map * _nRows * _nCols, _nRows * _nCols);
            }
            return ret;
        }

        public void BindTo(Layer nextLayer, bool randomizeWeights)
        {
            _nextLayer = nextLayer;
            nextLayer.BindInputFrom(this, randomizeWeights);
        }

        internal int N_Maps
        {
            get
            {
                return _nMaps;
            }
        }

        internal int N_Rows
        {
            get
            {
                return _nRows;
            }
        }

        internal int N_Cols
        {
            get
            {
                return _nCols;
            }
        }

        internal NeuronMap[] Maps
        {
            get
            {
                return _maps;
            }
        }
    }
}
