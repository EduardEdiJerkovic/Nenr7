package hr.fer.model;

public class Sample {
    private double[] point;
    private double[] classification;

    Sample(double[] point, double[] classification) {
        this.point = point;
        this.classification = classification;
    }

    public double[] getPoint() {
        return point;
    }

    public double[] getClassification() {
        return classification;
    }
}
