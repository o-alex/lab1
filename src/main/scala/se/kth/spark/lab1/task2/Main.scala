package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import scala.collection.mutable
object Main {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sparkContext.textFile(filePath).toDF("col")
    rawDF.show(1)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("col")
      .setOutputCol("features")
      .setPattern(",")

    println("\nTokenized values are ")
    //Step2: transform with tokenizer and show 5 rows
    var colsData = regexTokenizer.transform(rawDF)
    colsData.select("features").take(5).foreach(println)

    println("\nVectors are ")
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector().setInputCol("features").setOutputCol("vector")
    var vectors = arr2Vect.transform(colsData)
    vectors.select("vector").take(5).foreach(println)

    //Step4: extract the label(year) into a new column
    //using a udf. http://stackoverflow.com/questions/30219592/create-new-column-with-function-in-spark-dataframe
      import org.apache.spark.sql.functions._
      val coder: (Vector=> Double) = (arg: Vector) => arg.apply(0)
      val myudf = org.apache.spark.sql.functions.udf(coder)
      val lSlicer = vectors.withColumn("year", myudf(col("vector")))
      lSlicer.select("year").take(5).foreach(x => println(x.getClass.getSimpleName))

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(???)

    ???

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    val lShifter = new DoubleUDF(???)
    ???
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = ???

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(???)

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    ???

    //Step11: drop all columns from the dataframe other than label and features
    ???
  }
}