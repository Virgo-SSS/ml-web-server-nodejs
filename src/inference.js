// load tensorflow/tfjs-node package
const tf = require('@tensorflow/tfjs-node');

// function to load the model
function loadModel() {
  return tf.loadLayersModel('file://./models/model.json');
}

// function to predict the output
function predict(model , imageBuffer) {
    const tensor = tf.node
    .decodeJpeg(imageBuffer)
    .resizeNearestNeighbor([150,150])
    .expandDims()
    .toFloat();

    return model.predict(tensor).data();
}

module.exports = {
    loadModel,
    predict
};

