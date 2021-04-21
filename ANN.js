
async function PredictEnergyConsumption() {
  //These are the most common house features in the U.S.
  var ModelInput = {TYPEHUQ:2, DISHWASH:1, CWASHER:1,DRYER:1,AIRCOND:1,NUMBERAC:0,NUMCFAN:0,NUMFLOORFAN:0,NUMWHOLEFAN:0,NUMATTICFAN:0,NOTMOIST:0,FUELH20:1,USEEL:1,ELWARM:0,ELCOOL:1,ELWATER:0, ELFOOD:1,ELOTHER:1,TOTCSQFT:0,TOTHSQFT:0,TOTSQFT_EN:1152,HEATHOME:1,TVCOLOR:2,REGIONC_Northeast:0,REGIONC_Midwest:0,REGIONC_South:0,REGIONC_West:0};

  //Update with values input by the user
  var TOTSQFT_EN = document.getElementById('TOTSQFT_EN');
  var TOTCSQFT = document.getElementById('TOTCSQFT');
  var TOTHSQFT = document.getElementById('TOTHSQFT');
  var NUMBERAC = document.getElementById('NUMBERAC');
  var RegionOptions = document.getElementsByName('Region');

  ModelInput.TOTSQFT_EN = parseInt(TOTSQFT_EN.value)
  ModelInput.TOTCSQFT = parseInt(TOTCSQFT.value)
  ModelInput.TOTHSQFT = parseInt(TOTHSQFT.value)
  ModelInput.NUMBERAC = parseInt(NUMBERAC.value)
  if(RegionOptions[0].checked) ModelInput.REGIONC_Northeast = parseInt(1)
  if(RegionOptions[1].checked) ModelInput.REGIONC_Midwest = parseInt(1)
  if(RegionOptions[2].checked) ModelInput.REGIONC_South = parseInt(1)
  if(RegionOptions[3].checked) ModelInput.REGIONC_West = parseInt(1)

  //Convert the updated object into an tensor to feed into the model
  var input_array = Object.keys(ModelInput).map((key) => [ModelInput[key]]);
  let flattened = [].concat.apply([], input_array);

  var model_input_tensor = tf.tensor2d(flattened, [1,27]);

  // Script for applying the model. Guide from https://www.tensorflow.org/js/tutorials/conversion/import_keras and https://github.com/carlos-aguayo/carlos-aguayo.github.io/blob/master/tfjs.html 
  tf.loadLayersModel('model/model.json').then(function(model) {
      window.model = model;
      var prediction = window.model.predict(model_input_tensor);
      document.getElementById("tensorflow").innerHTML
      = "Predicted Energy Consumption: "+Math.round(prediction.arraySync()[0][0],2)+" KWH <br>";
    });

}