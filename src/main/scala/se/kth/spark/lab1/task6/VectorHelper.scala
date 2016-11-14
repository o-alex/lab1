package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    if( v1.size != v2.size) throw new IllegalArgumentException("Size of the input vectors does not match")

    var res:Double = 0
    for( i <- 0 to (v1.size -1 )){
      res += v1.apply(i) * v2.apply(i)
    }
    res
  }

  def dot(v: Vector, s: Double): Vector = {
    val arr = v.toArray
    arr.map(_*s)
    Vectors.dense(arr)
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    if( v1.size != v2.size) throw new IllegalArgumentException("Size of the input vectors does not match")
    var res:Array[Double] = new Array(v1.size)
    for( i <- 0 to (v1.size -1 )){
      res(i) = v1.apply(i) + v2.apply(i)
    }
    Vectors.dense(res)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    var arr:Array[Double] = new Array(size)
    arr.map(_=>fillVal)
    Vectors.dense(arr)
  }
}