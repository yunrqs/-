
import params from './params.mjs';

const util = {};


var resizeWidth = 10;
var resizeHeight = 6;

//not used !?
/**
 * Eye class, represents an eye patch detected in the video stream
 * @param {ImageData} patch - the image data corresponding to an eye
 * @param {Number} imagex - x-axis offset from the top-left corner of the video canvas
 * @param {Number} imagey - y-axis offset from the top-left corner of the video canvas
 * @param {Number} width  - width of the eye patch
 * @param {Number} height - height of the eye patch
 */
util.Eye = function(patch, imagex, imagey, width, height) {
    this.patch = patch;
    this.imagex = imagex;
    this.imagey = imagey;
    this.width = width;
    this.height = height;
};

/**
 * Compute eyes size as gray histogram
 * @param {Object} eyes - The eyes where looking for gray histogram
 * @returns {Array.<T>} The eyes gray level histogram
 */
util.getEyeFeats = function(eyes) {
    let process = (eye) => {
        let resized = this.resizeEye(eye, resizeWidth, resizeHeight);
        let gray = this.grayscale(resized.data, resized.width, resized.height);
        let hist = [];
        this.equalizeHistogram(gray, 5, hist);
        return hist;
    };
    if (params.trackEye == 'left') {
        return process(eyes.left);
    }
    else if (params.trackEye == 'right') {
        return process(eyes.right);
    }
    else {
        return [].concat(process(eyes.left), process(eyes.right));
    }
}

/**
 * Append all the contents of data
 * @param {Array} data - to be inserted
 */


//Helper functions
/**
 * Grayscales an image patch. Can be used for the whole canvas, detected face, detected eye, etc.
 *
 * Code from tracking.js by Eduardo Lundgren, et al.
 * https://github.com/eduardolundgren/tracking.js/blob/master/src/tracking.js
 *
 * Software License Agreement (BSD License) Copyright (c) 2014, Eduardo A. Lundgren Melo. All rights reserved.
 * Redistribution and use of this software in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * The name of Eduardo A. Lundgren Melo may not be used to endorse or promote products derived from this software without specific prior written permission of Eduardo A. Lundgren Melo.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @param  {Array} pixels - image data to be grayscaled
 * @param  {Number} width  - width of image data to be grayscaled
 * @param  {Number} height - height of image data to be grayscaled
 * @return {Array} grayscaledImage
 */
util.grayscale = function(pixels, width, height){
    var gray = new Uint8ClampedArray(pixels.length >> 2);
    var p = 0;
    var w = 0;
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var value = pixels[w] * 0.299 + pixels[w + 1] * 0.587 + pixels[w + 2] * 0.114;
            gray[p++] = value;

            w += 4;
        }
    }
    return gray;
};

/**
 * Increase contrast of an image.
 *
 * Code from Martin Tschirsich, Copyright (c) 2012.
 * https://github.com/mtschirs/js-objectdetect/blob/gh-pages/js/objectdetect.js
 *
 * @param {Array} src - grayscale integer array
 * @param {Number} step - sampling rate, control performance
 * @param {Array} dst - array to hold the resulting image
 */
util.equalizeHistogram = function(src, step, dst) {
    var srcLength = src.length;
    if (!dst) dst = src;
    if (!step) step = 5;

    // Compute histogram and histogram sum:
    var hist = Array(256).fill(0);

    for (var i = 0; i < srcLength; i += step) {
        ++hist[src[i]];
    }

    // Compute integral histogram:
    var norm = 255 * step / srcLength,
        prev = 0;
    for (var i = 0; i < 256; ++i) {
        var h = hist[i];
        prev = h += prev;
        hist[i] = h * norm; // For non-integer src: ~~(h * norm + 0.5);
    }

    // Equalize image:
    for (var i = 0; i < srcLength; ++i) {
        dst[i] = hist[src[i]];
    }
    return dst;
};

//not used !?
util.threshold = function(data, threshold) {
    for (let i = 0; i < data.length; i++) {
        data[i] = (data[i] > threshold) ? 255 : 0;
    }
    return data;
};

//not used !?
util.correlation = function(data1, data2) {
    const length = Math.min(data1.length, data2.length);
    let count = 0;
    for (let i = 0; i < length; i++) {
        if (data1[i] === data2[i]) {
            count++;
        }
    }
    return count / Math.max(data1.length, data2.length);
};

/**
 * Gets an Eye object and resizes it to the desired resolution
 * @param  {webgazer.util.Eye} eye - patch to be resized
 * @param  {Number} resizeWidth - desired width
 * @param  {Number} resizeHeight - desired height
 * @return {webgazer.util.Eye} resized eye patch
 */
util.resizeEye = function(eye, resizeWidth, resizeHeight) {

    var canvas = document.createElement('canvas');
    canvas.width = eye.width;
    canvas.height = eye.height;

    canvas.getContext('2d', { willReadFrequently: true }).putImageData(eye.patch,0,0);

    var tempCanvas = document.createElement('canvas');

    tempCanvas.width = resizeWidth;
    tempCanvas.height = resizeHeight;

    // save the canvas into temp canvas
    tempCanvas.getContext('2d').drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, resizeWidth, resizeHeight);

    return tempCanvas.getContext('2d').getImageData(0, 0, resizeWidth, resizeHeight);
};


export default util;

