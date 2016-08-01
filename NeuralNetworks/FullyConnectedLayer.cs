using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class FullyConnectedLayer : Layer1D
    {
        private WeightSet[] _weightSets;
        private int _activationFunction;
        private float _retentionRate;
        private float _retentionInverse;
        bool[] keep;

        public FullyConnectedLayer(int nNeurons, float retentionRate, int activationFunction)
            : base(nNeurons)
        {
            _neurons = new Neuron[nNeurons];
            _weightSets = new WeightSet[nNeurons];
            _activationFunction = activationFunction;
            _retentionRate = retentionRate;
            _retentionInverse = 1f / _retentionRate;
            keep = new bool[_nNeurons];
        }

        public FullyConnectedLayer(StreamReader reader)
        {
            _retentionRate = float.Parse(reader.ReadLine());
            _retentionInverse = 1f/_retentionRate;
            _activationFunction = int.Parse(reader.ReadLine());
            _nNeurons = int.Parse(reader.ReadLine());
            _weightSets = new WeightSet[_nNeurons];
            _neurons = new Neuron[_nNeurons];
            for (int i = 0; i < _nNeurons; i++)
                _weightSets[i] = WeightSet.GetWeightSetFrom(reader);
            keep = new bool[_nNeurons];
        }

        internal override void BindInputFrom(Layer prevLayer, bool randomizeWeights)
        {
            _prevLayer = prevLayer;
            Neuron[] prevLayerNeurons = prevLayer.GetAllNeurons();
            int prevLayerCount = prevLayerNeurons.Length;
            for (int i = 0; i < _nNeurons; i++)
            {
                if (randomizeWeights)
                    _weightSets[i] = new WeightSet(prevLayerCount, NeuralNetwork1D.Rand);
                _neurons[i] = new Neuron(prevLayerNeurons, _weightSets[i], _activationFunction);
            }
        }

        internal override float[] Propagate(bool realRun)
        {
            UpdateNeuronValues(realRun);
            return _nextLayer.Propagate(realRun);
        }

        internal override float PropagateAndLearn(float[] targetOutput, float learningRate)
        {
            UpdateNeuronValues(false);
            return _nextLayer.PropagateAndLearn(targetOutput, learningRate);
        }

        internal override void BackPropagate(float learningRate)
        {
            Parallel.For(0, _nNeurons, i =>
            {
                _neurons[i].Derivative *= _retentionInverse;
                _neurons[i].Value *= _retentionRate;
                _neurons[i].BackProp();
                _neurons[i].UpdateWeights(learningRate);
            });
            
            _prevLayer.BackPropagate(learningRate);
        }

        internal override void SaveWeights(StreamWriter writer)
        {
            writer.WriteLine(typeof(FullyConnectedLayer));
            writer.WriteLine(_retentionRate);
            writer.WriteLine(_activationFunction);
            writer.WriteLine(_nNeurons);
            for (int i = 0; i < _nNeurons; i++)
                _weightSets[i].SaveWeights(writer);
        }
        
        private void UpdateNeuronValues(bool realRun)
        {
            for (int i = 0; i < _nNeurons; i++) keep[i] = true;
            if (!realRun)
            {
                for (int i = 0; i < (1 - _retentionRate) * _nNeurons; i++)
                {
                    int k = NeuralNetwork2D.Rand.Next(_nNeurons);
                    while (!keep[k])
                        k = NeuralNetwork2D.Rand.Next(_nNeurons);
                    keep[k] = false;
                }
            }

            Parallel.For(0, _nNeurons, i =>
            {
                if (keep[i])
                {
                    _neurons[i].UpdateValue();
                    if (!realRun)
                        _neurons[i].Value *= _retentionInverse;
                }
                else _neurons[i].Die();
            });
        }
    }
}
