package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: org.apache.spark.ml.linalg.Vector, v2: org.apache.spark.ml.linalg.Vector): Double = {
    if( v1.size != v2.size) throw new IllegalArgumentException("Size of the input vectors does not match")

    var res:Double = 0
    for( i <- 0 to (v1.size -1 )){
      res += v1.apply(i) * v2.apply(i)
    }
    res
  }

  def dot(v: org.apache.spark.ml.linalg.Vector, s: Double): org.apache.spark.ml.linalg.Vector = {
    val arr = v.toArray
    Vectors.dense(arr.map(_*s))
  }

  def sum(v1: org.apache.spark.ml.linalg.Vector, v2: org.apache.spark.ml.linalg.Vector): org.apache.spark.ml.linalg.Vector = {
    if( v1.size != v2.size) throw new IllegalArgumentException("Size of the input vectors does not match")
    var res:Array[Double] = new Array(v1.size)
    for( i <- 0 to (v1.size -1 )){
      res(i) = v1.apply(i) + v2.apply(i)
    }
    Vectors.dense(res)
  }

  def fill(size: Int, fillVal: Double): org.apache.spark.ml.linalg.Vector = {
    var arr:Array[Double] = new Array(size)
    Vectors.dense(arr.map(_=>fillVal))
  }
}