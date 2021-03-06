package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.hack._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.mllib.evaluation.RegressionMetrics

case class Instance(label: Double, features: org.apache.spark.ml.linalg.Vector)

object Helper {
  def rmse(labelsAndPreds: RDD[(Double, Double)]): Double = {
    val v = labelsAndPreds.map(x=> Math.pow(x._1-x._2,2)).reduce(_+_)
    v/labelsAndPreds.count()
    scala.math.sqrt(v/labelsAndPreds.count())
  }

  def predictOne(weights: org.apache.spark.ml.linalg.Vector, features: org.apache.spark.ml.linalg.Vector): Double = {
    if( weights.size != features.size) throw new IllegalArgumentException("Size of the input vectors does not match")
    var res : Double = 0
    for( i <- 0 to (weights.size -1 )){
     res += weights.apply(i) * features.apply(i)
    }
    res
  }

  def predict(weights: org.apache.spark.ml.linalg.Vector, data: RDD[Instance]): RDD[(Double, Double)] = {
    data.map(d=>(d.label, predictOne(weights, d.features)))
  }
}

class MyLinearRegressionImpl(override val uid: String)
    extends MyLinearRegression[Vector, MyLinearRegressionImpl, MyLinearModelImpl] {

  def this() = this(Identifiable.randomUID("mylReg"))

  override def copy(extra: ParamMap): MyLinearRegressionImpl = defaultCopy(extra)

  def gradientSummand(weights: Vector, lp: Instance): Vector = {
    val pred = Helper.predictOne(weights,lp.features)
    val actual = lp.label
    VectorHelper.dot(lp.features, (pred-actual))
  }

  def gradient(d: RDD[Instance], weights: Vector): Vector = {
    d.map(x=>gradientSummand(weights,x)).reduce((x,y)=>VectorHelper.sum(x,y))
  }

  def linregGradientDescent(trainData: RDD[Instance], numIters: Int): (Vector, Array[Double]) = {

    val n = trainData.count()
    println("Training Data Size is "+n)
    val d = trainData.take(1)(0).features.size

//    var weights = VectorHelper.fill(d, random.nextDouble())

    val random = scala.util.Random
    var arr:Array[Double] = new Array(d)
    for(i <- 0 until d){
     arr(i) = random.nextDouble()
    }
    var weights = Vectors.dense(arr)


    val alpha = 0.54
    val errorTrain = Array.fill[Double](numIters)(0)

    for (i <- 0 until numIters) {
      //compute this iterations set of predictions based on our current weights
      val labelsAndPredsTrain = Helper.predict(weights, trainData)
      //compute this iteration's RMSE
      errorTrain(i) = Helper.rmse(labelsAndPredsTrain)
      println("RMSE for round "+i+" is: "+errorTrain(i))

      //compute gradient
      val g = gradient(trainData, weights)
      //update the gradient step - the alpha
//      val alpha_i = alpha / (n * scala.math.sqrt(i + 1))
      val alpha_i = alpha / (n )
      val wAux = VectorHelper.dot(g, (-1) * alpha_i)
      //update weights based on gradient
      weights = VectorHelper.sum(weights, wAux)
    }
    (weights, errorTrain)
  }

  def train(dataset: Dataset[_]): MyLinearModelImpl = {
    println("Training")

    val numIters = 100

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          Instance(label, features)
      }

    val (weights, trainingError) = linregGradientDescent(instances, numIters)
    new MyLinearModelImpl(uid, weights, trainingError)
  }
}

class MyLinearModelImpl(override val uid: String, val weights: Vector, val trainingError: Array[Double])
    extends MyLinearModel[Vector, MyLinearModelImpl] {

  override def copy(extra: ParamMap): MyLinearModelImpl = defaultCopy(extra)

  def predict(features: Vector): Double = {
    println("Predicting")
    val prediction = Helper.predictOne(weights, features)
    prediction
  }
}