package hr.fer.model;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class GeneticAlgorithm {

    private List<NeuralNetwork> population = new ArrayList<>();
    private final int maxIterations;
    private final double epsilon;
    private final INeuralNetworkEvaluator nne;
    private final Random r = new Random();
    private double[] crossoverProbab;

    public GeneticAlgorithm(int populationSize, int[] configuration, INeuralNetworkEvaluator nne, int maxIterations, double epsilon) {
        this.maxIterations = maxIterations;
        this.epsilon = epsilon;
        this.nne = nne;
        for (int i = 0; i < populationSize; ++i) {
            population.add(new NeuralNetwork(configuration));
        }
        population.forEach(nn -> nn.evaluateFitness(nne));
    }

    public NeuralNetwork run() {
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            population.sort((nn1, nn2) -> (int) Math.signum(nn1.getFitness() - nn2.getFitness()));
            if (population.get(0).getFitness() < epsilon) {
                System.out.println("Iteration: " + iteration);
                return population.get(0);
            }
            if (iteration % (maxIterations / population.size()) == 0) {
                System.out.println("Iteration" + iteration + ": " + population.get(0).getFitness());
            }
            List<Integer> parentIndexes = pickParents(3);
            Collections.sort(parentIndexes);
            NeuralNetwork child = crossover(population.get(parentIndexes.get(0)), population.get(parentIndexes.get(1)));
            child = mutation(child);
            child.evaluateFitness(nne);
            population.set(parentIndexes.get(2), child);
        }
        population.forEach(nn -> nn.evaluateFitness(nne));
        return population.get(0);
    }

    public NeuralNetwork crossover(NeuralNetwork nn1, NeuralNetwork nn2) {
        int option = r.nextInt(3);
        switch (option) {
            case 0:
                return crossover1(nn1, nn2);
            case 1:
                return crossover2(nn1, nn2);
            case 2:
                return crossover3(nn1, nn2);
            default:
                return null;
        }
    }

    /***
     * Crossover function with algorithm that given 50% chance for every gene to come from one of two parents.
     *
     * @param nn1 First parent.
     * @param nn2 Second parent.
     * @return New neural network (child) of two given parents.
     */
    private NeuralNetwork crossover1(NeuralNetwork nn1, NeuralNetwork nn2) {
        var c = new NeuralNetwork(nn1.getConfiguration());
        double[][][] w1 = nn1.getWeights();
        double[][][] w2 = nn2.getWeights();
        double[][][] b1 = nn1.getBiases();
        double[][][] b2 = nn2.getBiases();
        for (int i = 0; i < nn1.getConfiguration().length - 1; ++i) {
            for (int j = 0; j < nn1.getConfiguration()[i + 1]; ++j) {
                for (int k = 0; k < w1[i][j].length; ++k) {
                    if (r.nextInt(2) == 1) {
                        c.getWeights()[i][j][k] = w2[i][j][k];
                    } else {
                        c.getWeights()[i][j][k] = w1[i][j][k];
                    }
                }
                for (int k = 0; k < b1[i][j].length; ++k) {
                    if (r.nextInt(2) == 1) {
                        c.getBiases()[i][j][k] = b2[i][j][k];
                    } else {
                        c.getBiases()[i][j][k] = b1[i][j][k];
                    }
                }
            }
        }
        return c;
    }

    /***
     * Crossover function with algorithm that given 50% chance for every node to come from one of two parents.
     *
     * @param nn1 First parent.
     * @param nn2 Second parent.
     * @return New neural network (child) of two given parents.
     */
    private NeuralNetwork crossover2(NeuralNetwork nn1, NeuralNetwork nn2) {
        var c = new NeuralNetwork(nn1.getConfiguration());
        double[][][] w1 = nn1.getWeights();
        double[][][] w2 = nn2.getWeights();
        double[][][] b1 = nn1.getBiases();
        double[][][] b2 = nn2.getBiases();
        for (int i = 0; i < nn1.getConfiguration().length - 1; ++i) {
            for (int j = 0; j < nn1.getConfiguration()[i + 1]; ++j) {
                if (r.nextInt(2) == 1) {
                    c.getWeights()[i][j] = w2[i][j].clone();
                    c.getBiases()[i][j] = b2[i][j].clone();
                } else {
                    c.getWeights()[i][j] = w1[i][j].clone();
                    c.getBiases()[i][j] = b1[i][j].clone();
                }
            }
        }
        return c;
    }

    private NeuralNetwork crossover3(NeuralNetwork nn1, NeuralNetwork nn2) {
        var c = new NeuralNetwork(nn1.getConfiguration());
        double[][][] w1 = nn1.getWeights();
        double[][][] w2 = nn2.getWeights();
        double[][][] b1 = nn1.getBiases();
        double[][][] b2 = nn2.getBiases();
        for (int i = 0; i < nn1.getConfiguration().length - 1; ++i) {
            for (int j = 0; j < nn1.getConfiguration()[i + 1]; ++j) {
                for (int k = 0; k < w1[i][j].length; ++k) {
                    double a = r.nextDouble();
                    c.getWeights()[i][j][k] = (1 - a) * w1[i][j][k] + a * w2[i][j][k];
                }
                for (int k = 0; k < b1[i][j].length; ++k) {
                    double a = r.nextGaussian();
                    c.getBiases()[i][j][k] = (1 - a) * b1[i][j][k] + a * b2[i][j][k];
                }
            }
        }
        return c;
    }

    public NeuralNetwork mutation(NeuralNetwork nn) {
        int option = r.nextInt(2);
        switch (option) {
            case 0:
                return mutation1(nn);
            case 1:
                return mutation2(nn);
            default:
                return null;
        }
    }

    private NeuralNetwork mutation1(NeuralNetwork nn) {
        double probability = 0.01;
        for (int i = 0; i < nn.getConfiguration().length - 1; ++i) {
            for (int j = 0; j < nn.getConfiguration()[i + 1]; ++j) {
                for (int k = 0; k < nn.getWeights()[i][j].length; ++k) {
                    if (r.nextDouble() < probability) {
                        nn.getWeights()[i][j][k] = r.nextGaussian() * NeuralNetwork.WEIGHT_VARIANCE;
                    }
                }
                if (nn.getBiases()[i][j].length > 1) {
                    for (int k = 0; k < nn.getBiases()[i][j].length; ++k) {
                        if (r.nextDouble() < probability) {
                            nn.getBiases()[i][j][k] = r.nextGaussian() * NeuralNetwork.SCALE_VARIANCE;
                        }
                    }
                } else {
                    nn.getBiases()[i][j][0] = r.nextGaussian() * NeuralNetwork.BIAS_VARIANCE;
                }
            }
        }
        return nn;
    }

    private NeuralNetwork mutation2(NeuralNetwork nn) {
        double probability = 0.01;
        for (int i = 0; i < nn.getConfiguration().length - 1; ++i) {
            for (int j = 0; j < nn.getConfiguration()[i + 1]; ++j) {
                if (r.nextDouble() < probability) {
                    for (int k = 0; k < nn.getWeights()[i][j].length; ++k) {
                        nn.getWeights()[i][j][k] = r.nextGaussian() * NeuralNetwork.WEIGHT_VARIANCE * 5;
                    }
                    if (nn.getBiases()[i][j].length > 1) {
                        for (int k = 0; k < nn.getBiases()[i][j].length; ++k) {
                            nn.getBiases()[i][j][k] = r.nextGaussian() * NeuralNetwork.SCALE_VARIANCE * 5;
                        }
                    } else {
                        nn.getBiases()[i][j][0] = r.nextGaussian() * NeuralNetwork.BIAS_VARIANCE * 5;
                    }
                }
            }
        }
        return nn;
    }

    private List<Integer> pickParents(int size) {
        List<Integer> indexes = new ArrayList<>();
        while (indexes.size() != size) {
            var tmp = r.nextInt(population.size());
            if (indexes.contains(tmp)) {
                continue;
            } else {
                indexes.add(tmp);
            }
        }

        return indexes;

        /*var result = new NeuralNetwork[size];
        for (int i = 0; i < size; ++i) {
            result[i] = this.population.get(indexes.get(i));
        }

        return result;*/
    }
}
