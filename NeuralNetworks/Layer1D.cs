using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public abstract class Layer1D : Layer
    {
        protected Neuron[] _neurons;
        protected int _nNeurons;

        protected Layer1D()
        {
        }

        protected Layer1D(int nNeurons)
        {
            _nNeurons = nNeurons;
        }

        internal override Neuron[] GetAllNeurons()
        {
            return _neurons;
        }

        public void BindTo(Layer1D nextLayer, bool randomizeWeights)
        {
            _nextLayer = nextLayer;
            nextLayer.BindInputFrom(this, randomizeWeights);
        }
    }
}
