package hr.fer.model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Samples implements INeuralNetworkEvaluator {

    private List<Sample> samples = new ArrayList<>();
    private final int numberOfClasses;

    public Samples(String path, int numberOfClasses) throws IOException {
        this.numberOfClasses = numberOfClasses;
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            while (line != null) {
                String[] values = line.split("\t");
                double[] point = new double[values.length - this.numberOfClasses];
                double[] classification = new double[numberOfClasses];
                for (int i = 0; i < values.length; ++i) {
                    if (i >= values.length - numberOfClasses) {
                        classification[i - point.length] = Double.parseDouble(values[i]);
                    } else {
                        point[i] = Double.parseDouble(values[i]);
                    }
                }
                samples.add(new Sample(point, classification));
                line = br.readLine();
            }
        }
    }

    public Sample getSample(int index) {
        if (index < 0 || index > samples.size()) {
            return null;
        }
        return this.samples.get(index);
    }

    public int getSaplesSize() {
        return samples.size();
    }

    public double errorEvaluation(NeuralNetwork nn) {
        double error = 0.0;
        for (Sample s : this.samples) {
            double[] nnPrediction = nn.valueOf(s.getPoint());
            for (int i = 0; i < nnPrediction.length; ++i) {
                error += Math.pow(nnPrediction[i] - s.getClassification()[i], 2);
            }
        }
        return error / this.samples.size();
    }

    public int prediction(NeuralNetwork nn) {
        int predictionCount = 0;
        for (var s : this.samples) {
            int[] result = new int[this.numberOfClasses];
            var value = nn.valueOf(s.getPoint());
            for (int i = 0; i < nn.valueOf(s.getPoint()).length; ++i) {
                result[i] = value[i] < 0.5 ? 0 : 1;
            }
            for (int i = 0; i < result.length; ++i) {

            }
            var print = s.toString();
            print += "prediction: ";
            for (var r : result) {
                print += r + "\t";
            }
            boolean prediction = true;
            for (int i = 0; i < numberOfClasses; ++i) {
                if (result[i] != s.getIntClassification()[i]) {
                    prediction = false;
                    break;
                }
            }
            print += "predict: ";
            if (prediction) {
                predictionCount++;
                print += "True";
            } else {
                print += "False";
            }
            System.out.println(print);
        }
        return predictionCount;
    }
}
