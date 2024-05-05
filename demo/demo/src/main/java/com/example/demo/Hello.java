package com.example.demo;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Map;

import org.apache.commons.math3.stat.regression.AbstractMultipleLinearRegression;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import it.unimi.dsi.fastutil.Arrays;

import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

@RestController
public class Hello {
    @RequestMapping("/hello")
    public String hello() {
        return "Hello";
    }
    private SavedModelBundle model;

    public void loadModel(String modelPath) {
        model = SavedModelBundle.load(modelPath, "serve");
    }

    public Tensor<?> predict(Tensor<?> inputTensor) {
        try (Session session = model.session()) {
            Tensor<?> outputTensor = session.runner()
                .feed("input_tensor_name", inputTensor)
                .fetch("output_tensor_name")
                .run()
                .get(0);
            return outputTensor;
        }
    }

    public void close() {
        model.close();
    }

    @RequestMapping("/predict")
    public String modelPrediction(@RequestBody double[] features) {
        // Convert input features to INDArray
        INDArray input = Nd4j.create(features);

        // Make prediction
        INDArray output = model.output(input);
        // Convert output to double array
        double[] outputArray = output.toDoubleVector();

        // Convert double array to string representation
        String outputString = java.util.Arrays.toString(outputArray);

        return outputString;
    }

    
}
