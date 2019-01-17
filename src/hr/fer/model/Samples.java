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

    public double errorEvaluation(NeuralNetwork nn) {
        double error = 0.0;
        for (Sample s : this.samples) {
            double[] nnPrediction = nn.valueOf(s.getPoint());
            for (int i = 0; i < nnPrediction.length; ++i) {
                error += Math.pow(nnPrediction[i] - s.getClassification()[i], 2);
            }
        }
        return error;
    }
}
