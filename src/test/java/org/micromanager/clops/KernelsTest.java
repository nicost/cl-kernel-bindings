package org.micromanager.clops;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import net.haesleinhuepf.clij.clearcl.ClearCL;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLContext;
import net.haesleinhuepf.clij.clearcl.ClearCLDevice;
import net.haesleinhuepf.clij.clearcl.ClearCLImage;
import net.haesleinhuepf.clij.clearcl.backend.ClearCLBackendInterface;
import net.haesleinhuepf.clij.clearcl.backend.ClearCLBackends;
import net.haesleinhuepf.clij.clearcl.enums.HostAccessType;
import net.haesleinhuepf.clij.clearcl.enums.ImageChannelDataType;
import net.haesleinhuepf.clij.clearcl.enums.KernelAccessType;
import net.haesleinhuepf.clij.clearcl.enums.MemAllocMode;
import net.haesleinhuepf.clij.clearcl.exceptions.OpenCLException;
import net.haesleinhuepf.clij.clearcl.ocllib.OCLlib;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.coremem.offheap.OffHeapMemory;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.micromanager.clops.CLKernelException;
import org.micromanager.clops.CLKernelExecutor;
import org.micromanager.clops.Kernels;

/**
 *
 * @author nico
 */
public class KernelsTest {

   private ClearCLContext gCLContext;
   private CLKernelExecutor gCLKE;
   final long xSize = 1024;
   final long ySize = 1024;
   final long zSize = 4;
   final long[] dimensions1D
           = {xSize * ySize};
   final long[] dimensions2D
           = {xSize, ySize};
   final long[] dimensions3D
           = {xSize, ySize, zSize};
   final long[][] allDimensions
           = {dimensions1D, dimensions2D, dimensions3D};
   ClearCLImage srcFloat, srcUByte, srcUShort;
   ClearCLImage dstFloat, dstUByte, dstUShort;
   ClearCLBuffer srcBufFloat, srcBufUByte, srcBufUShort;
   ClearCLBuffer dstBufFloat, dstBufUByte, dstBufUShort;
   ClearCLImage[] srcImages;
   ClearCLImage[] dstImages;
   ClearCLBuffer[] srcBuffers
           = {srcBufFloat, srcBufUByte, srcBufUShort};
   ClearCLBuffer[] dstBuffers
           = {dstBufFloat, dstBufUByte, dstBufUShort};
   ClearCLImage dstFloat3D;

   @Before
   public void initKernelTests() throws IOException {
      ClearCLBackendInterface lClearCLBackend
              = ClearCLBackends.getBestBackend();

      ClearCL lClearCL = new ClearCL(lClearCLBackend);

      // initialisation with the first device found
      // ClearCLDevice lBestGPUDevice = lClearCL.getAllDevices().get(0);
      ClearCLDevice lBestGPUDevice = lClearCL.getBestGPUDevice();

      gCLContext = lBestGPUDevice.createContext();

      gCLKE = new CLKernelExecutor(gCLContext, OCLlib.class);

      // create src and dst images and buffers for all types to speed up testing
      // and reduce testing code
      try {
         srcBufFloat = gCLKE.createCLBuffer(dimensions2D,
                 NativeTypeEnum.Float);
         dstBufFloat = gCLKE.createCLBuffer(srcBufFloat);
         srcBufUByte = gCLKE.createCLBuffer(dimensions2D,
                 NativeTypeEnum.UnsignedByte);
         dstBufUByte = gCLKE.createCLBuffer(srcBufUByte);
         srcBufUShort = gCLKE.createCLBuffer(dimensions2D,
                 NativeTypeEnum.UnsignedShort);
         dstBufUShort = gCLKE.createCLBuffer(srcBufUShort);
       } catch (OpenCLException cle) {
         Assert.fail(cle.getMessage());
      } finally {
         srcBuffers = new ClearCLBuffer[]{srcBufFloat, srcBufUByte, srcBufUShort};
         dstBuffers = new ClearCLBuffer[]{dstBufFloat, dstBufUByte, dstBufUShort};
      }
      
      try {         
         srcFloat = gCLKE.createCLImage(dimensions2D,
                 ImageChannelDataType.Float);
         dstFloat = gCLKE.createCLImage(srcFloat);
         srcUByte = gCLKE.createCLImage(dimensions2D,
                 ImageChannelDataType.UnsignedInt8);
         dstUByte = gCLKE.createCLImage(srcUByte);
         srcUShort
                 = gCLKE.createCLImage(dimensions2D,
                         ImageChannelDataType.UnsignedInt16);
         dstUShort = gCLKE.createCLImage(srcUShort);  
         dstFloat3D = gCLKE.createCLImage(dimensions3D,
                 ImageChannelDataType.Float); 
      } catch (OpenCLException cle) {
         System.out.println("Context: " + lBestGPUDevice.getName() + " does not support CLImages");
      } finally {
         srcImages = new ClearCLImage[]{srcFloat, srcUByte, srcUShort};
         dstImages = new ClearCLImage[]{dstFloat, dstUByte, dstUShort};
      }

   }

   @After
   public void cleanupKernelTests() throws IOException {
      for (ClearCLImage clImg : srcImages) {
         if (clImg != null) {
            clImg.close();
         }
      }
      for (ClearCLImage clImg : dstImages) {
         if (clImg != null) {
            clImg.close();
         }
      }
      for (ClearCLBuffer clBuf : srcBuffers) {
         if (clBuf != null) {
            clBuf.close();
         }
      }
      for (ClearCLBuffer clBuf : dstBuffers) {
         if (clBuf != null) {
            clBuf.close();
         }
      }

      gCLKE.close();

      gCLContext.close();
   }

   @Test
   public void testAbsolute() throws IOException {
      // Todo: check unsigned integer types?
      try {
         if (srcFloat != null && dstFloat != null) {
            Kernels.set(gCLKE, srcFloat, -3.0f);
            Kernels.absolute(gCLKE, srcFloat, dstFloat);
            float[] minMax = Kernels.minMax(gCLKE, dstFloat, 36);
            Assert.assertEquals(3.0f, minMax[0], 0.000001);
         }
         Kernels.set(gCLKE, srcBufFloat, -5.0f);
         Kernels.absolute(gCLKE, srcBufFloat, dstBufFloat);
         float[] minMax = Kernels.minMax(gCLKE, dstBufFloat, 36);
         Assert.assertEquals(5.0f, minMax[0], 0.000001);
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testAddImages() throws IOException {
      try {
         for (int i = 0; i < srcImages.length; i++) {
            if (srcImages[i] != null) {
               float[] minMax;
               try (ClearCLImage src2 = gCLKE.createCLImage(srcImages[i])) {
                  Kernels.set(gCLKE, srcImages[i], 1.0f);
                  Kernels.set(gCLKE, src2, 2.0f);
                  Kernels.addImages(gCLKE, srcImages[i], src2, dstImages[i]);
                  minMax = Kernels.minMax(gCLKE, dstImages[i], 36);
               }
               Assert.assertEquals(3.0f, minMax[0], 0.000001);
            }
         }
         for (int i = 0; i < srcBuffers.length; i++) {
            float[] minMax;
            try (ClearCLBuffer src2 = gCLKE.createCLBuffer(srcBuffers[i])) {
               Kernels.set(gCLKE, srcBuffers[i], 1.0f);
               Kernels.set(gCLKE, src2, 2.0f);
               Kernels.addImages(gCLKE, srcBuffers[i], src2, dstBuffers[i]);
               minMax = Kernels.minMax(gCLKE, dstBuffers[i], 36);
            }
            Assert.assertEquals(3.0f, minMax[0], 0.000001);
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }

   }

   @Test
   public void testAddImagesWeighted() throws IOException {
      try {
         final float x = 3.0f;
         final float y = 7.0f;
         final float a = 2.0f;
         final float b = 3.0f;
         // ensure this will still work with integers
         final float result = x * a + y * b;
         for (int i = 0; i < srcImages.length; i++) {
            if (srcImages[i] != null) {
               float[] minMax;
               try (ClearCLImage src2 = gCLKE.createCLImage(srcImages[i])) {
                  Kernels.set(gCLKE, srcImages[i], x);
                  Kernels.set(gCLKE, src2, y);
                  Kernels.addImagesWeighted(gCLKE,
                          srcImages[i],
                          src2,
                          dstImages[i],
                          a,
                          b);
                  minMax = Kernels.minMax(gCLKE, dstImages[i], 36);
               }
               Assert.assertEquals(result, minMax[0], 0.0000001);
            }
         }
         for (int i = 0; i < srcBuffers.length; i++) {
            float[] minMax;
            try (ClearCLBuffer src2 = gCLKE.createCLBuffer(srcBuffers[i])) {
               Kernels.set(gCLKE, srcBuffers[i], x);
               Kernels.set(gCLKE, src2, y);
               Kernels.addImagesWeighted(gCLKE,
                       srcBuffers[i],
                       src2,
                       dstBuffers[i],
                       a,
                       b);
               minMax = Kernels.minMax(gCLKE, dstBuffers[i], 36);
            }
            Assert.assertEquals(result, minMax[0], 0.000001);
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testAddImageAndScalar() throws IOException {
      try {
         for (int i = 0; i < srcImages.length; i++) {
            if (srcImages[i] != null) {
               Kernels.set(gCLKE, srcImages[i], 1.0f);
               Kernels.addImageAndScalar(gCLKE,
                       srcImages[i],
                       dstImages[i],
                       4.0f);
               float minMax[] = Kernels.minMax(gCLKE, dstImages[i], 36);
               Assert.assertEquals(5.0f, minMax[0], 0.0000001);
            }
         }
         for (int i = 0; i < srcBuffers.length; i++) {
            Kernels.set(gCLKE, srcBuffers[i], 11.0f);
            Kernels.addImageAndScalar(gCLKE,
                    srcBuffers[i],
                    dstBuffers[i],
                    -3.0f);
            float minMax[] = Kernels.minMax(gCLKE, dstBuffers[i], 36);
            Assert.assertEquals(8.0f, minMax[0], 0.000001);
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testArgMaximumZProjection() {
      try {
         Kernels.xorFractal(gCLKE, dstFloat3D, 2, 3, 0.2f);
         try (ClearCLImage dstFloatIndex = gCLKE.createCLImage(dstFloat)) {
            Kernels.argMaximumZProjection(gCLKE,
                    dstFloat3D,
                    dstFloat,
                    dstFloatIndex);
            float[] minMax = Kernels.minMax(gCLKE, dstFloat, 36);
            Assert.assertEquals(409.4, minMax[1], 0.0001);
            minMax = Kernels.minMax(gCLKE, dstFloatIndex, 36);
            Assert.assertEquals(3.0f, minMax[1], 0.0001);
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }

   }

   @Test
   public void testBlurImage() {
      try {
         for (int i = 0; i < srcBuffers.length; i++) {
            Kernels.blur(gCLKE, srcBuffers[i], dstBuffers[i], 4.0f, 4.0f);
         }
         for (int i = 0; i < srcImages.length; i++) {
            if (srcImages[i] != null) {
               Kernels.blur(gCLKE, srcImages[i], dstImages[i], 4.0f, 4.0f);
            }
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testBinaryAnd() {
      try {
         for (int i = 0; i < srcBuffers.length; i++) {
            try (ClearCLBuffer mask = gCLKE.createCLBuffer(srcBuffers[i])) {
               Kernels.set(gCLKE, mask, 1.0f);
               Kernels.binaryAnd(gCLKE, srcBuffers[i], mask, dstBuffers[i]);
               // Check equality between src and dst
               Kernels.subtractImages(gCLKE,
                       srcBuffers[i],
                       dstBuffers[i],
                       mask);
               float[] minMax = Kernels.minMax(gCLKE, mask, 36);
               Assert.assertEquals(minMax[0], minMax[1], 0.0000001);
            }
         }
         for (int i = 0; i < srcImages.length; i++) {
            if (srcImages[i] != null) {
               try (ClearCLImage mask = gCLKE.createCLImage(srcImages[i])) {
                  Kernels.binaryAnd(gCLKE, srcImages[i], mask, dstImages[i]);
                  Kernels.set(gCLKE, mask, 1.0f);
                  // Check equality between src and dst
                  Kernels.subtractImages(gCLKE,
                          srcBuffers[i],
                          dstBuffers[i],
                          mask);
                  float[] minMax = Kernels.minMax(gCLKE, mask, 36);
                  Assert.assertEquals(minMax[0], minMax[1], 0.0000001);
               }
            }
         }
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testMinMaxBuffer() {
      for (long[] lDimensions : allDimensions) {
         try (ClearCLBuffer lCLBuffer = gCLContext.createBuffer(MemAllocMode.Best,
                 HostAccessType.ReadWrite,
                 KernelAccessType.ReadWrite,
                 1,
                 NativeTypeEnum.Float,
                 lDimensions)) {
            OffHeapMemory lBuffer
                    = OffHeapMemory.allocateFloats(lCLBuffer.getLength());
            float lJavaMin = Float.POSITIVE_INFINITY;
            float lJavaMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lCLBuffer.getLength(); i++) {
               float lValue = 1f / (1f + i);
               lJavaMin = Math.min(lJavaMin, lValue);
               lJavaMax = Math.max(lJavaMax, lValue);
               lBuffer.setFloatAligned(i, lValue);
            }
            lCLBuffer.readFrom(lBuffer, true);
            try {
               float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLBuffer, 128);
               assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
               assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

            } catch (CLKernelException clkExc) {
               Assert.fail(clkExc.getMessage());
            }

         }
      }

   }

   @Test
   public void testMinMaxImageFloat() {

      try (ClearCLImage lCLImage = gCLKE.createCLImage(dimensions2D,
              ImageChannelDataType.Float)) {
         long size = lCLImage.getWidth() * lCLImage.getHeight();
         OffHeapMemory lBuffer = OffHeapMemory.allocateFloats(size);
         float lJavaMin = Float.POSITIVE_INFINITY;
         float lJavaMax = Float.NEGATIVE_INFINITY;
         for (int i = 0; i < size; i++) {
            float lValue = 1f / (1f + i);
            lJavaMin = Math.min(lJavaMin, lValue);
            lJavaMax = Math.max(lJavaMax, lValue);
            lBuffer.setFloatAligned(i, lValue);
         }
         lCLImage.readFrom(lBuffer, true);
         try {
            float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
            assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
            assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

         } catch (CLKernelException clkExc) {
            Assert.fail(clkExc.getMessage());
         }

      }
   }

   @Test
   public void testMinMaxImageUI16() {

      try (ClearCLImage lCLImage = gCLKE.createCLImage(dimensions2D,
              ImageChannelDataType.UnsignedInt16)) {
         long size = lCLImage.getWidth() * lCLImage.getHeight();
         OffHeapMemory lBuffer = OffHeapMemory.allocateShorts(size);
         float lJavaMin = Float.POSITIVE_INFINITY;
         float lJavaMax = Float.NEGATIVE_INFINITY;
         for (int i = 0; i < size; i++) {
            float lValue = 23000f / (1f + i) + 129.0f;
            lJavaMin = (int) Math.min(lJavaMin, lValue);
            lJavaMax = (int) Math.max(lJavaMax, lValue);
            short sv = (short) (0xFFFF & (int) lValue);
            lBuffer.setShortAligned(i, sv);
         }
         lCLImage.readFrom(lBuffer, true);
         try {
            float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
            assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
            assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

         } catch (CLKernelException clkExc) {
            Assert.fail(clkExc.getMessage());
         }

      }
   }

   @Test
   public void testMinMaxImageUI8() {

      try (ClearCLImage lCLImage = gCLKE.createCLImage(dimensions2D,
              ImageChannelDataType.UnsignedInt8)) {
         long size = lCLImage.getWidth() * lCLImage.getHeight();
         OffHeapMemory lBuffer = OffHeapMemory.allocateBytes(size);
         float lJavaMin = Float.POSITIVE_INFINITY;
         float lJavaMax = Float.NEGATIVE_INFINITY;
         for (int i = 0; i < size; i++) {
            float lValue = 220.0f / (1f + i) + 5.0f;
            lJavaMin = (int) Math.min(lJavaMin, lValue);
            lJavaMax = (int) Math.max(lJavaMax, lValue);
            byte sv = (byte) (0xFF & ((int) lValue));
            lBuffer.setByteAligned(i, sv);
         }
         lCLImage.readFrom(lBuffer, true);
         try {
            float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
            assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
            assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

         } catch (CLKernelException clkExc) {
            Assert.fail(clkExc.getMessage());
         }

      }
   }

   @Test
   public void testMinimumImages() {
      ImageChannelDataType[] types
              = {ImageChannelDataType.Float,
                 ImageChannelDataType.UnsignedInt16,
                 ImageChannelDataType.UnsignedInt8};
      for (ImageChannelDataType type : types) {
         testMinimumImages(type);
      }
   }

   public void testMinimumImages(ImageChannelDataType type) {
      ClearCLImage src1 = gCLKE.createCLImage(dimensions2D, type);
      ClearCLImage src2 = gCLKE.createCLImage(src1);
      ClearCLImage dst = gCLKE.createCLImage(src1);

      try {
         Kernels.set(gCLKE, src1, 3.0f);
         Kernels.set(gCLKE, src2, 1.0f);
         Kernels.minimumImages(gCLKE, src1, src2, dst);
         // TODO: test that src2 and dst are identical

      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

   @Test
   public void testHistogram() {
      try (ClearCLImage lCLImage = gCLKE.createCLImage(dimensions2D,
              ImageChannelDataType.UnsignedInt16)) {
         long size = lCLImage.getWidth() * lCLImage.getHeight();
         OffHeapMemory lBuffer = OffHeapMemory.allocateShorts(size);
         float lJavaMin = Float.POSITIVE_INFINITY;
         float lJavaMax = Float.NEGATIVE_INFINITY;
         for (int i = 0; i < size; i++) {
            float lValue = 23000f / (1f + i) + 129.0f;
            lJavaMin = (int) Math.min(lJavaMin, lValue);
            lJavaMax = (int) Math.max(lJavaMax, lValue);
            short sv = (short) (0xFFFF & (int) lValue);
            lBuffer.setShortAligned(i, sv);
         }
         lCLImage.readFrom(lBuffer, true);
         try {
            // OpenCL uses uShort to index the histogram
            // however, other limits - possibly hardware related - are reached
            // at smaller size already. For now, just be sure that sizes other
            // than 256 actually work
            int[] histLengths = {256, 2048};
            for (int histLength : histLengths) {
               // CPU histogram calculation
               int[] cpuHist = new int[histLength];
               int min = (int) lJavaMin;
               int max = (int) lJavaMax;
               int range = max - min;
               int maxIndex = histLength - 1;
               float histLengthDivRange = (float) histLength / (float) range;

               for (long i = 0; i < size; i++) {
                  short val = lBuffer.getShortAligned(i);
                  int iVal = 0xFFFF & val;
                  int index = (int) ((iVal - min) * histLengthDivRange);
                  index = index > maxIndex ? maxIndex : index;
                  cpuHist[index]++;
               }
               // GPU histogram calculation
               float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
               int[] gpuHist = new int[histLength];
               Kernels.histogram(gCLKE,
                       lCLImage,
                       gpuHist,
                       lOpenCLMinMax[0],
                       lOpenCLMinMax[1]);
               // check they are equal.  Because of possible differences in rounding,
               // accept a difference of 1
               long sum = 0;
               for (int i = 0; i < gpuHist.length; i++) {
                  sum += gpuHist[i];
                  // System.out.println(" " + i + ": " + " CPU: " + cpuHist[i] + ", GPU: " + gpuHist[i]);
                  assertEquals(gpuHist[i], cpuHist[i], 1.0);
               }
               assertEquals(size, sum, 1.0);
            }
         } catch (CLKernelException clkExc) {
            Assert.fail(clkExc.getMessage());
         }
      }
   }

   @Test
   public void testPhantomImages() throws IOException {
      try {
         Kernels.xorFractal(gCLKE, dstFloat, 2, 3, 0.2f);
         float[] minMax = Kernels.minMax(gCLKE, dstFloat, 36);
         Assert.assertEquals(0.0, minMax[0], 0.0000001);
         Assert.assertEquals(409.4, minMax[1], 0.00001);
         Kernels.xorFractal(gCLKE, dstUShort, 2, 3, 0.2f);
         minMax = Kernels.minMax(gCLKE, dstUShort, 36);
         Assert.assertEquals(0.0f, minMax[0], 0.0000001);
         Assert.assertEquals(409.0f, minMax[1], 0.00000001);
         Kernels.xorSphere(gCLKE, dstFloat3D, 0, 0, 0, 40.0f);
         Kernels.sphere(gCLKE, dstFloat3D, 0, 0, 0, 40.0f);
         Kernels.line(gCLKE, dstFloat3D, 5, 5, 25, 25, 2.0f);
      } catch (CLKernelException clkExc) {
         Assert.fail(clkExc.getMessage());
      }
   }

}
