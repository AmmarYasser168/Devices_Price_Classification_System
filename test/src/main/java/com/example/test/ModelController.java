package com.example.test;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Map;


import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;



import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

@RestController
public class ModelController {
    @RequestMapping("/hello")
    public String hello() {
        return "Hello";
    }
    private SavedModelBundle model;
    public ModelController(){
        loadModel("E:\\Ammar\\University\\Machine Learning\\Devices_Price_Classification_System\\ANN_model");
    }
    public void loadModel(String modelPath) {
        model = SavedModelBundle.load(modelPath, "serve");
    }
    @RequestMapping("/predict")
    public Tensor predict(@RequestBody Tensor inputTensor) {
        try (Session session = model.session()) {
            Tensor outputTensor = session.runner()
                .feed("X", inputTensor)
                .fetch("Y")
                .run()
                .get(0);
            return outputTensor;
        }
    }

    public void close() {
        model.close();
    }

    
}
