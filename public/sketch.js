var cats;
var rainbows;
var trains;
var clocks
var catTrainingDatax=[];
var trainTrainingDatax=[];
var rainbowTrainingDatax=[];
var clockTrainingDatax=[];
var imageSize=1000;
var img;
var tfmodel;
function preload(){
	cats=loadBytes("cat1000.bin");
	rainbows=loadBytes("rainbow1000.bin");
	trains=loadBytes("train1000.bin");
	clocks=loadBytes("clock.npy");
}
function createData(){
	var index=0;
	for(var i=0;i<cats.bytes.length/784;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(cats.bytes[index]/255);
			index++;
		}
		catTrainingDatax.push(row);
	}
	index=0;
	for(var i=0;i<trains.bytes.length/784;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(trains.bytes[index]/255);
			index++;
		}
		trainTrainingDatax.push(row);
	}
	index=0;
	for(var i=0;i<rainbows.bytes.length/784;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(rainbows.bytes[index]/255);
			index++;
		}
		rainbowTrainingDatax.push(row);
	}
	index=80;
	for(var i=0;i<1000;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(clocks.bytes[index]/255);
			index++;
		}
		clockTrainingDatax.push(row);
	}
	
}
function setup(){
	createCanvas(280,280);
	background(0);
	createData();
	// createtfmodel();
	// train()
	test();
	// code to diaply a image
	// img=createImage(28,28);
	// img.loadPixels();
	// var index=0;
	// for(var i=0;i<784;i++){
	// 	img.pixels[index+0]=clockTrainingDatax[0][i];
	// 	img.pixels[index+1]=clockTrainingDatax[0][i];
	// 	img.pixels[index+2]=clockTrainingDatax[0][i];
	// 	img.pixels[index+3]=255;
	// 	index+=4;
	// }
	// img.updatePixels();
	// img.resize(280,280);
	// image(img,0,0);
}
function createtfmodel(){
	tfmodel = tf.sequential();
    tfmodel.add(tf.layers.conv2d({
      inputShape: [28,28,1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));
    tfmodel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
    tfmodel.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));
    tfmodel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
    tfmodel.add(tf.layers.flatten());
    tfmodel.add(tf.layers.dense({
      units: 4,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }));
    // const LEARNING_RATE = 0.15;
    // const optimizer = tf.train.sgd(LEARNING_RATE);
    tfmodel.compile({
      optimizer: "rmsprop",
      loss: 'meanSquaredError',
      metrics: ['accuracy'],
	});
}
async function train(){
	var iterations=50;
	var trainx=[];
	var trainy=[];
	for(var j=0;j<800;j++){
		trainx.push(catTrainingDatax[j]);
		trainy.push([1,0,0,0]);
		trainx.push(trainTrainingDatax[j]);
		trainy.push([0,1,0,0]);
		trainx.push(rainbowTrainingDatax[j]);
		trainy.push([0,0,1,0]);
		trainx.push(clockTrainingDatax[j]);
		trainy.push([0,0,0,1]);
		
	}
	var x=tf.tensor(trainx,[3200,28,28,1]);
	var y=tf.tensor(trainy,[3200,4]);
	for(var i=0;i<iterations;i++){
		var options={
			validationData:null,
			epochs:1,
			shuffle:true
		}
		var result=await tfmodel.fit(x,y,options);
		console.log("Epochs= "+i+" Loss= "+result.history.loss[0]);
	}
	console.log("Training Done");
	tfmodel.save('downloads://doodlemodel');
}
async function predict(){
	var tfmodel=await tf.loadLayersModel('doodlemodel.json');
	console.log("Model loaded");
	var testx=[];
	var img=get();
	img.resize(28,28);
	img.loadPixels();
	var index=0;
	for(var i=0;i<img.pixels.length;i+=4){
		testx[index]=img.pixels[i]/255;
		index++;
	}
	var x=tf.tensor([testx],[1,28,28,1]);
	// var x=tf.tensor(trainTrainingDatax[900],[1,28,28,1]);
	var result=tfmodel.predict(x);
	result.print();
	document.getElementById("result").innerHTML=result
}
function find_max(a){
	var  max=a[0];
	var  max_index=0;
	for(var i=1;i<a.length;i++){
			if(a[i]>max){
				max=a[i];
				max_index=i;
			}
	}
	return max_index;
}
async function test(){
	tfmodel=await tf.loadLayersModel('doodlemodel.json');
	console.log("Model loaded");
	var total=0;
	var correct=0;
	for(var j=800;j<1000;j++){
		var resultTensor=tfmodel.predict(tf.tensor(catTrainingDatax[j],[1,28,28,1]))
		var result=await resultTensor.data();
		var max_index=find_max(result);
		if(max_index==0){
			correct++;
		}
		total++;
		resultTensor=tfmodel.predict(tf.tensor(trainTrainingDatax[j],[1,28,28,1]))
		result=await resultTensor.data();
		max_index=find_max(result);
		if(max_index==1){
			correct++;
		}
		total++;
		resultTensor=tfmodel.predict(tf.tensor(rainbowTrainingDatax[j],[1,28,28,1]))
		result=await resultTensor.data();
		max_index=find_max(result);
		if(max_index==2){
			correct++;
		}
		total++;
	}
	console.log("Accuracy= "+((correct/total)*100)+"%");
}
function clearScreen(){
	background(0);
}
function draw(){
	strokeWeight(8);
	stroke(255);
	if(mouseIsPressed){
		line(pmouseX,pmouseY,mouseX,mouseY);
	}
}