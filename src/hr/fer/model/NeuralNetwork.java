package hr.fer.model;

import java.util.Random;

public class NeuralNetwork {
    private double[][] cache;
    private double[][][] weights;
    private double[][][] biases;
    private double fitness;
    private int[] configuration;

    private static Random r = new Random();
    public static final double WEIGHT_VARIANCE = 100.0;
    public static final double BIAS_VARIANCE = 10.0;
    public static final double SCALE_VARIANCE = 10.0;

    public NeuralNetwork(int... configuration) {
        this.configuration = configuration;
        this.cache = new double[configuration.length][];
        for (int i = 0; i < configuration.length; ++i) {
            this.cache[i] = new double[configuration[i]];
        }
        this.weights = new double[configuration.length - 1][][];
        this.biases = new double[configuration.length - 1][][];
        for (int i = 0; i < configuration.length - 1; ++i) {
            this.weights[i] = new double[configuration[i + 1]][];
            this.biases[i] = new double[configuration[i + 1]][];
            for (int j = 0; j < configuration[i + 1]; ++j) {
                this.weights[i][j] = new double[configuration[i]];
                for (int k = 0; k < configuration[i]; ++k) {
                    this.weights[i][j][k] = r.nextGaussian() * WEIGHT_VARIANCE;
                }
                if (i == 0) {
                    this.biases[i][j] = new double[configuration[i]];
                    for (int k = 0; k < this.biases[i][j].length; k++) {
                        this.biases[i][j][k] = r.nextGaussian() * SCALE_VARIANCE;
                    }
                } else {
                    this.biases[i][j] = new double[1];
                    for (int k = 0; k < this.biases[i][j].length; k++) {
                        this.biases[i][j][k] = r.nextGaussian() * BIAS_VARIANCE;
                    }
                }
            }
        }
    }

    public NeuralNetwork(int[] configuration, double[][][] weights, double[][][] biases) {
        this.weights = weights;
        this.biases = biases;
        for (int i = 0; i < configuration.length; ++i) {
            this.cache[i] = new double[configuration[i]];
        }
        this.configuration = configuration;
    }

    public double getFitness() {
        return fitness;
    }

    public void evaluateFitness(INeuralNetworkEvaluator nne) {
        this.fitness = nne.errorEvaluation(this);
    }

    public int[] getConfiguration() {
        return configuration;
    }

    public double[][][] getWeights() {
        return weights;
    }

    public double[][][] getBiases() {
        return biases;
    }

    public double[] valueOf(double... point) {
        for (int i = 0; i < point.length; ++i) {
            this.cache[0][i] = point[i];
        }
        for (int i = 0; i < this.weights.length; ++i) {
            if (i == 0) {
                for (int j = 0; j < this.weights[i].length; ++j) {
                    //var cache = this.cache[i + 1][j];
                    this.cache[i + 1][j] = 0;
                    for (int k = 0; k < this.weights[i][j].length; ++k) {
                        this.cache[i + 1][j] += Math.abs(this.cache[i][k] - this.weights[i][j][k]) / this.biases[i][j][k];
                    }
                    this.cache[i + 1][j] = 1 / (1 + this.cache[i + 1][j]);
                }
            } else {
                for (int j = 0; j < this.weights[i].length; ++j) {
                    this.cache[i + 1][j] = 0;
                    for (int k = 0; k < this.weights[i][j].length; ++k) {
                        this.cache[i + 1][j] += this.cache[i][k] * this.weights[i][j][k];
                    }
                    this.cache[i + 1][j] += this.biases[i][j][0];
                    this.cache[i + 1][j] = 1 / (1 + Math.exp(-this.cache[i + 1][j]));
                }
            }
        }
        return this.cache[this.weights.length];
    }

    public String toString() {
        var result = "";
        return result;
    }
}
