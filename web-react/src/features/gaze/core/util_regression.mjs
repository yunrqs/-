import mat from './mat.mjs';
const util_regression = {};

/**
 * Performs ridge regression, according to the Weka code.
 * @param {Array} y - corresponds to screen coordinates (either x or y) for each of n click events
 * @param {Array.<Array.<Number>>} X - corresponds to gray pixel features (120 pixels for both eyes) for each of n clicks
 * @param {Array} k - ridge parameter
 * @return{Array} regression coefficients
 */
util_regression.ridge = function(y, X, k){
    var nc = X[0].length;
    var m_Coefficients = new Array(nc);
    var xt = mat.transpose(X);
    var solution = new Array();
    var success = true;
    var attempts = 0;
    do{
        if (++attempts > 8) throw new Error('校准矩阵无法求解，请重新校准');
        var ss = mat.mult(xt,X);
        // Set ridge regression adjustment
        for (var i = 0; i < nc; i++) {
            ss[i][i] = ss[i][i] + k;
        }

        // Carry out the regression
        var bb = mat.mult(xt,y);
        for(var i = 0; i < nc; i++) {
            m_Coefficients[i] = bb[i][0];
        }
        try{
            solution = mat.solve(ss, bb);

            for (var i = 0; i < nc; i++){
                m_Coefficients[i] = solution[i][0];
            }
            if (!m_Coefficients.every(Number.isFinite)) throw new Error('Invalid regression coefficients');
            success = true;
        }
        catch (ex){
            k *= 10;
            console.log(ex);
            success = false;
        }
    } while (!success);
    return m_Coefficients;
}


export default util_regression;

