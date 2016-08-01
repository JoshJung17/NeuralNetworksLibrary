using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    internal class WeightSet
    {
        private float[] _weights;
        private float _bias;
        private int _nInput;
        private bool _modifiable;

        internal WeightSet(int nInput, Random rand)
        {
            _nInput = nInput;
            _weights = new float[nInput];
            RandomizeWeights(nInput, rand);
            _modifiable = true;
        }

        internal WeightSet(float[] weights, float bias, bool modifiable)
        {
            _nInput = weights.Length;
            _weights = weights;
            _bias = bias;
            _modifiable = modifiable;
        }

        internal float EvaluateWith(Neuron[] inputNeurons)
        {
            float sum = _bias;
            for (int i = 0; i < _nInput; i++)
            {
                sum += inputNeurons[i].Value * _weights[i];
            }
            return sum;
        }

        internal void BackProp(Neuron[] inputNeurons, float derivative)
        {
            for (int i = 0; i < _nInput; i++)
                inputNeurons[i].Derivative += derivative * _weights[i];
        }

        internal void UpdateWeights(Neuron[] inputNeurons, float derivative, float learningRate)
        {
            if (!_modifiable) return;
            _bias -= derivative * learningRate;
            for (int i = 0; i < _nInput; i++)
                _weights[i] -= derivative * inputNeurons[i].Value * learningRate;
        }

        private void RandomizeWeights(int nInput, Random rand)
        {
            double stdDev = 1.0 / Math.Sqrt(nInput);
            for (int i = 0; i < nInput; i++)
            {
                _weights[i] = GetRandomGaussian(stdDev, rand);
            }
        }

        private float GetRandomGaussian(double stdDev, Random rand)
        {
            double d1 = rand.NextDouble(), d2 = rand.NextDouble();
            return (float)(stdDev * Math.Sqrt(-2.0 * Math.Log(d1)) * Math.Sin(2.0 * Math.PI * d2));
        }

        internal void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(_modifiable);
            writer.WriteLine(_bias);
            writer.WriteLine(_nInput);
            for (int i = 0; i < _nInput; i++)
                writer.WriteLine(_weights[i]);
        }

        internal static WeightSet GetWeightSetFrom(StreamReader reader)
        {
            bool modifiable = bool.Parse(reader.ReadLine());
            float bias = float.Parse(reader.ReadLine());
            int nInput = int.Parse(reader.ReadLine());
            float[] weights = new float[nInput];
            for (int i = 0; i < nInput; i++)
                weights[i] = float.Parse(reader.ReadLine());
            return new WeightSet(weights, bias, modifiable);
        }
    }
}
