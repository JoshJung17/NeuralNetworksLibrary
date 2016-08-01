using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class InputLayer : Layer1D
    {
        public InputLayer(int nNeurons)
            : base(nNeurons)
        {
            _neurons = new Neuron[nNeurons];
            for (int i = 0; i < nNeurons; i++)
            {
                _neurons[i] = new Neuron();
            }
        }

        public InputLayer(StreamReader reader)
        {
            _nNeurons = int.Parse(reader.ReadLine());
            _neurons = new Neuron[_nNeurons];
            for (int i = 0; i < _nNeurons; i++)
            {
                _neurons[i] = new Neuron();
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
            writer.WriteLine(typeof(InputLayer));
            writer.WriteLine(_nNeurons);
        }

        public float[] RunNetwork(float[] input, bool realRun)
        {
            for (int i = 0; i < _nNeurons; i++) _neurons[i].Value = input[i];
            return _nextLayer.Propagate(realRun);
        }

        public float Train(float[] input, float[] targetOutput, float learningRate)
        {
            for (int i = 0; i < _nNeurons; i++) _neurons[i].Value = input[i];
            
            return _nextLayer.PropagateAndLearn(targetOutput, learningRate);
        }
    }
}
