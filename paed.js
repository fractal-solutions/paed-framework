import NeuralNetwork from "./nn"
// Initialize parameters
const parameters = {
    alpha: 0.005, // Trend energy weight
    beta: 0.005, // Volatility energy weight
    gamma: 0.0001, // Liquidity flow weight
    eta: 0.0001, // Learning rate for price updates
    noiseThreshold: 0.01, // Threshold to filter noise
    stopLossMultiplier: 2 // Multiplier for volatility-based stop loss
};
  
// Data structure to hold price action data
const marketData = {
    prices: [], // Array to store price data
    volumes: [], // Array to store volume data
    timeSteps: 0 // Number of data points
};
  
// Helper function to calculate the local average price
function calculateLocalAverage(prices, windowSize) {
    const movingAverages = [];
    for (let i = 0; i < prices.length; i++) {
      const window = prices.slice(Math.max(0, i - windowSize + 1), i + 1);
      const average = window.reduce((a, b) => a + b, 0) / window.length;
      movingAverages.push(average);
    }
    return movingAverages;
}
  

// Calculate Trend Energy
function calculateTrendEnergy(prices) {
    let trendEnergy = 0;
    for (let t = 1; t < prices.length; t++) {
      trendEnergy -= parameters.alpha * (prices[t] - prices[t - 1]);
    }
    return trendEnergy;
}
  
// Calculate Volatility Energy
function calculateVolatilityEnergy(prices, localAverages) {
    let volatilityEnergy = 0;
    for (let t = 0; t < prices.length; t++) {
      volatilityEnergy += parameters.beta * Math.pow(prices[t] - localAverages[t], 2);
    }
    return volatilityEnergy;
}
  
// Calculate Liquidity Flow
function calculateLiquidityFlow(prices, volumes) {
    let liquidityFlow = 0;
    for (let t = 1; t < prices.length; t++) {
      const priceChange = prices[t] - prices[t - 1];
      liquidityFlow -= parameters.gamma * volumes[t] * priceChange;
    }
    return liquidityFlow;
}
  
// Update price to minimize energy
function updatePrices(prices, volumes, iterations = 100) {
    let updatedPrices = [...prices];
    for (let iter = 0; iter < iterations; iter++) {
      const localAverages = calculateLocalAverage(updatedPrices, 5);
      for (let t = 1; t < updatedPrices.length; t++) {
        const trend = -parameters.alpha * (updatedPrices[t] - updatedPrices[t - 1]);
        const volatility = parameters.beta * (updatedPrices[t] - localAverages[t]);
        const liquidity = -parameters.gamma * volumes[t] * (updatedPrices[t] - updatedPrices[t - 1]);
        const energyGradient = trend + volatility + liquidity;
        updatedPrices[t] -= parameters.eta * energyGradient;
      }
    }
    return updatedPrices;
}

// Learn and encode patterns
function encodePatterns(prices, patterns) {
    const weights = Array(prices.length).fill().map(() => Array(prices.length).fill(0));
    for (const pattern of patterns) {
      for (let i = 0; i < pattern.length; i++) {
        for (let j = 0; j < pattern.length; j++) {
          weights[i][j] += (pattern[i] - prices[i]) * (pattern[j] - prices[j]);
        }
      }
    }
    return weights;
}

function temporalWeights(prices, decayRate = 0.1) {
    const weights = [];
    for (let i = 0; i < prices.length; i++) {
        weights[i] = [];
        for (let j = 0; j < prices.length; j++) {
            weights[i][j] = Math.exp(-decayRate * Math.abs(i - j));
        }
    }
    return weights;
}

function momentumWeights(prices, timeStep = 1) {
    const weights = [];

    // Step 1: Calculate raw weights
    for (let i = 0; i < prices.length; i++) {
        weights[i] = [];
        for (let j = 0; j < prices.length; j++) {
            weights[i][j] = (prices[j] - prices[i]) / (timeStep * Math.abs(j - i + 1));
        }
    }

    return weights;
}

// Generate signals based on stable patterns
function generateSignalsX(prices, weights) {
    const signals = [];
    for (let t = 0; t < prices.length; t++) {
      const signalStrength = weights[t].reduce((sum, w) => sum + w * prices[t], 0);
      if (signalStrength > parameters.noiseThreshold) {
        signals.push({ time: t, type: "BUY", strength: signalStrength });
      }
    }
    return signals;
}

// Generate signals based on stable patterns
function generateSignals(prices, weights) {
    const signals = [];
    for (let t = 0; t < prices.length; t++) {
        const signalStrength = weights[t].reduce((sum, w) => sum + w * prices[t], 0);
        
        // Define thresholds for buy and sell signals
        const buyThreshold = parameters.noiseThreshold; // Adjust as needed
        const sellThreshold = -parameters.noiseThreshold; // Adjust as needed for sell signals

        // Generate buy signal
        if (signalStrength > buyThreshold) {
            signals.push({ time: t, type: "BUY", strength: signalStrength });
        }
        // Generate sell signal
        else if (signalStrength < sellThreshold) {
            signals.push({ time: t, type: "SELL", strength: signalStrength });
        }
    }
    return signals;
}
  
// Set dynamic stop-loss based on volatility
function calculateStopLoss(price, volatility) {
    return price - parameters.stopLossMultiplier * Math.sqrt(volatility);
}

// Full PAED Trading Framework
function PAEDFramework(data) {
    // Extract data
    const { prices, volumes } = data;
    const localAverages = calculateLocalAverage(prices, 5);
    
    // Calculate components of the energy function
    const trendEnergy = calculateTrendEnergy(prices);
    const volatilityEnergy = calculateVolatilityEnergy(prices, localAverages);
    const liquidityFlow = calculateLiquidityFlow(prices, volumes);
    
    // Update prices
    const minimizedPrices = updatePrices(prices, volumes);
    
    
    // Encode patterns and generate signals
    //const weights = encodePatterns(prices, [[...prices.slice(-3)]]); // last 10 points as a pattern
    const weights = momentumWeights(prices)
    const signals = generateSignals(minimizedPrices, weights);

    const minimizedPricesX = minimizedPrices//.slice(-15);
    const pricesX = prices//.slice(-15);
    const signalsX = signals.slice(-15);
    
    return {
      trendEnergy,
      volatilityEnergy,
      liquidityFlow,
      minimizedPricesX,
      pricesX,
      signalsX
    };
}
  

// Function to extract close prices and volumes from JSON data
function extractDataFromJSON(jsonData, n) {
    // Check if jsonData and jsonData.data are defined
    if (!jsonData) {
        throw new Error("Invalid JSON data: 'data' property is missing.");
    }
    const data = JSON.parse(jsonData);
    const closeArr = [];
    const volumeArr = [];
    const {close,volume} = data;

    for (let i = 0; i < close.length; i++) {
        closeArr.push(close[i]);
        volumeArr.push(volume[i]);
    }


    const prices = closeArr.slice(-n); // Get last n close prices
    const volumes = volumeArr.slice(-n); // Get last n volumes
    return { prices, volumes };
}

// Example Usage
const fs = require('fs');
const jsonData = fs.readFileSync('data/EURUSD_H1.json', 'utf8');// Adjust the path as necessary
const n = 5000; // Specify how many last prices and volumes to retrieve
const data = extractDataFromJSON(jsonData, n);
const {
    trendEnergy,
    volatilityEnergy,
    liquidityFlow,
    minimizedPricesX,
    pricesX,
    signalsX
} = PAEDFramework(data);

let nn_inputs = [], nn_outputs = [];
{
    // Calculate winrate for BUY and SELL strategies without streaks
    let buyWins = 0;
    let buyLosses = 0;
    let sellWins = 0;
    let sellLosses = 0;
    let averageWinPips = 0;
    let averageLossPips = 0;

    const barCount = 5;
    for (let i = 0; i < minimizedPricesX.length - barCount; i++) {
        const mP = minimizedPricesX[i];
        const P = pricesX[i];
        const nextP = pricesX[i + barCount];
        const pipDifference = (nextP - P) * 10000; // Assuming prices are in a format where 1 pip = 0.0001

        nn_inputs.push([P, mP, P - mP]);

        let signal;
        if (mP > P) { // SELL signal
            signal = 'SELL';
            if (nextP < P) {
                sellWins++;
                averageWinPips += Math.sqrt(pipDifference * pipDifference);
            } else {
                sellLosses++;
                averageLossPips += Math.sqrt(pipDifference * pipDifference);
            }
        } else if (mP < P) { // BUY signal
            signal = 'BUY';
            if (nextP > P) {
                buyWins++;
                averageWinPips += Math.sqrt(pipDifference * pipDifference);
            } else {
                buyLosses++;
                averageLossPips += Math.sqrt(pipDifference * pipDifference);
            }
        }

        nn_outputs.push(signal === null ? 0 : 1);
    }



    const totalBuyTrades = buyWins + buyLosses;
    const totalSellTrades = sellWins + sellLosses;
    averageWinPips /= (buyWins + sellWins);
    averageLossPips /= (buyLosses + sellLosses);

    const buyWinRate = totalBuyTrades > 0 ? (buyWins / totalBuyTrades) * 100 : 0;
    const buyLossRate = totalBuyTrades > 0 ? (buyLosses / totalBuyTrades) * 100 : 0;
    const sellWinRate = totalSellTrades > 0 ? (sellWins / totalSellTrades) * 100 : 0;
    const sellLossRate = totalSellTrades > 0 ? (sellLosses / totalSellTrades) * 100 : 0;

    const expectancy = (buyWinRate / 100 * averageWinPips) - (buyLossRate / 100 * averageLossPips);
    
    console.log(`Buy Win Rate: ${buyWinRate.toFixed(2)}%`);
    console.log(`Sell Win Rate: ${sellWinRate.toFixed(2)}%`);
    console.log(`Avg Win: ${averageWinPips} pips || Avg Loss: ${averageLossPips} pips`);
    console.log(`Expectancy: ${expectancy.toFixed(2)} pips per trade`);
    
}

// const ratio = 0.8;
// const trainingData = nn_inputs.slice(0, Math.floor(nn_inputs.length * ratio)).map((input, index) => {
//     return { input: input, output: [nn_outputs[index]] };
// });

// const testData = nn_inputs.slice(Math.floor(nn_inputs.length * ratio)).map((input, index) => {
//     return { input: input, output: [nn_outputs[Math.floor(nn_inputs.length * ratio) + index]] };
// });

// const network = new NeuralNetwork(3,[8,8],2,'relu');
// const learningRate = 0.1;
// const epochs = 300;
// network.train(trainingData,learningRate,epochs,true);
// for(let input of testData){
//     const prediction = network.forward(input);
//     console.log(`Input: ${input}, Prediction: ${prediction}`);
// }

console.log({
    trendEnergy,
    volatilityEnergy,
    liquidityFlow,
    minimizedPricesX,
    pricesX,
    signalsX
});

  
  
  
