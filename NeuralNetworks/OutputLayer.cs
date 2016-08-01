using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class OutputLayer : Layer1D
    {
        public OutputLayer()
        {
        }

        public OutputLayer(StreamReader reader)
        {
        }

        internal override void BindInputFrom(Layer prevLayer, bool randomizeWeights)
        {
            _prevLayer = prevLayer;
            _neurons = prevLayer.GetAllNeurons();
            _nNeurons = _neurons.Length;
        }

        internal override float[] Propagate(bool realRun)
        {
            float[] ret = new float[_nNeurons];
            for (int i = 0; i < _nNeurons; i++)
            {
                ret[i] = _neurons[i].Value;
            }
            return ret;
        }

        internal override float PropagateAndLearn(float[] targetOutput, float learningRate)
        {
            return RunBackProp(targetOutput, learningRate);
        }

        internal float RunBackProp(float[] targetOutput, float learningRate)
        {
            float error = 0;
            for (int i = 0; i < _nNeurons; i++)
            {
                _neurons[i].Derivative = _neurons[i].Value - targetOutput[i];
                error += _neurons[i].Derivative * _neurons[i].Derivative;
            }

            _prevLayer.BackPropagate(learningRate);

            return error;
        }

        internal override void BackPropagate(float learningRate)
        {
            throw new NotImplementedException();
        }

        internal override void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(typeof(OutputLayer));
        }
    }
}
