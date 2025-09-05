const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Configuration for different networks
const NETWORK_CONFIG = {
  hardhat: {
    verificationThreshold: 70,
    minimumVerifiers: 2,
    submissionFee: ethers.utils.parseEther("0.001"),
    verificationReward: ethers.utils.parseEther("0.0005"),
  },
  ganache: {
    verificationThreshold: 70,
    minimumVerifiers: 2,
    submissionFee: ethers.utils.parseEther("0.001"),
    verificationReward: ethers.utils.parseEther("0.0005"),
  },
  goerli: {
    verificationThreshold: 75,
    minimumVerifiers: 3,
    submissionFee: ethers.utils.parseEther("0.01"),
    verificationReward: ethers.utils.parseEther("0.005"),
  },
  sepolia: {
    verificationThreshold: 75,
    minimumVerifiers: 3,
    submissionFee: ethers.utils.parseEther("0.01"),
    verificationReward: ethers.utils.parseEther("0.005"),
  },
  mumbai: {
    verificationThreshold: 75,
    minimumVerifiers: 3,
    submissionFee: ethers.utils.parseEther("0.01"),
    verificationReward: ethers.utils.parseEther("0.005"),
  },
  polygon: {
    verificationThreshold: 80,
    minimumVerifiers: 5,
    submissionFee: ethers.utils.parseEther("10"), // 10 MATIC
    verificationReward: ethers.utils.parseEther("2"), // 2 MATIC
  },
  mainnet: {
    verificationThreshold: 85,
    minimumVerifiers: 7,
    submissionFee: ethers.utils.parseEther("0.1"),
    verificationReward: ethers.utils.parseEther("0.02"),
  },
};

// Initial data sources for different networks
const INITIAL_DATA_SOURCES = {
  development: [
    {
      name: "Bloomberg ESG",
      credibilityScore: 95,
    },
    {
      name: "Refinitiv ESG",
      credibilityScore: 93,
    },
    {
      name: "MSCI ESG",
      credibilityScore: 92,
    },
    {
      name: "Sustainalytics",
      credibilityScore: 90,
    },
  ],
  production: [
    {
      name: "Bloomberg ESG",
      credibilityScore: 95,
    },
    {
      name: "Refinitiv ESG",
      credibilityScore: 93,
    },
    {
      name: "MSCI ESG",
      credibilityScore: 92,
    },
    {
      name: "Sustainalytics",
      credibilityScore: 90,
    },
    {
      name: "S&P Global ESG",
      credibilityScore: 88,
    },
    {
      name: "CDP",
      credibilityScore: 85,
    },
    {
      name: "SASB",
      credibilityScore: 87,
    },
  ],
};

async function main() {
  console.log("🚀 Starting ESG Data Oracle deployment...");
  
  // Get network information
  const network = await ethers.provider.getNetwork();
  const networkName = network.name === "unknown" ? "hardhat" : network.name;
  console.log(`📡 Deploying to network: ${networkName} (Chain ID: ${network.chainId})`);
  
  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log(`👤 Deploying with account: ${deployer.address}`);
  
  // Check deployer balance
  const balance = await deployer.getBalance();
  console.log(`💰 Account balance: ${ethers.utils.formatEther(balance)} ETH`);
  
  // Get network configuration
  const config = NETWORK_CONFIG[networkName] || NETWORK_CONFIG.hardhat;
  console.log(`⚙️  Network configuration:`, {
    verificationThreshold: config.verificationThreshold,
    minimumVerifiers: config.minimumVerifiers,
    submissionFee: ethers.utils.formatEther(config.submissionFee),
    verificationReward: ethers.utils.formatEther(config.verificationReward),
  });
  
  try {
    // Deploy ESGDataOracle contract
    console.log("\n📄 Deploying ESGDataOracle contract...");
    const ESGDataOracle = await ethers.getContractFactory("ESGDataOracle");
    
    // Estimate gas for deployment
    const deploymentData = ESGDataOracle.getDeployTransaction();
    const estimatedGas = await ethers.provider.estimateGas(deploymentData);
    console.log(`⛽ Estimated gas for deployment: ${estimatedGas.toString()}`);
    
    // Deploy contract
    const esgOracle = await ESGDataOracle.deploy({
      gasLimit: estimatedGas.mul(120).div(100), // Add 20% buffer
    });
    
    console.log(`⏳ Deployment transaction hash: ${esgOracle.deployTransaction.hash}`);
    console.log("⏳ Waiting for deployment confirmation...");
    
    await esgOracle.deployed();
    
    console.log(`✅ ESGDataOracle deployed to: ${esgOracle.address}`);
    console.log(`🧾 Deployment transaction: ${esgOracle.deployTransaction.hash}`);
    
    // Wait for a few confirmations
    console.log("⏳ Waiting for confirmations...");
    await esgOracle.deployTransaction.wait(2);
    
    // Configure the contract
    console.log("\n⚙️  Configuring contract...");
    
    // Update contract parameters if different from defaults
    if (config.verificationThreshold !== 70) {
      console.log(`🔧 Setting verification threshold to ${config.verificationThreshold}%`);
      const tx1 = await esgOracle.updateVerificationThreshold(config.verificationThreshold);
      await tx1.wait();
    }
    
    if (!config.submissionFee.eq(ethers.utils.parseEther("0.01"))) {
      console.log(`🔧 Setting submission fee to ${ethers.utils.formatEther(config.submissionFee)} ETH`);
      const tx2 = await esgOracle.updateSubmissionFee(config.submissionFee);
      await tx2.wait();
    }
    
    if (!config.verificationReward.eq(ethers.utils.parseEther("0.001"))) {
      console.log(`🔧 Setting verification reward to ${ethers.utils.formatEther(config.verificationReward)} ETH`);
      const tx3 = await esgOracle.updateVerificationReward(config.verificationReward);
      await tx3.wait();
    }
    
    // Register initial data sources
    console.log("\n📊 Registering initial data sources...");
    const isProduction = ["mainnet", "polygon"].includes(networkName);
    const dataSources = INITIAL_DATA_SOURCES[isProduction ? "production" : "development"];
    
    // For demo purposes, we'll use the deployer as the initial data sources
    // In production, these would be actual data provider addresses
    for (let i = 0; i < dataSources.length; i++) {
      const source = dataSources[i];
      // Generate a deterministic address for each data source
      const sourceWallet = ethers.Wallet.createRandom();
      
      console.log(`📝 Registering ${source.name} (${sourceWallet.address})`);
      const tx = await esgOracle.registerDataSource(
        sourceWallet.address,
        source.name,
        source.credibilityScore
      );
      await tx.wait();
    }
    
    // Add additional verifiers for production networks
    if (isProduction) {
      console.log("\n👥 Adding additional verifiers for production...");
      // In production, these would be actual verifier addresses
      const additionalVerifiers = [
        ethers.Wallet.createRandom().address,
        ethers.Wallet.createRandom().address,
        ethers.Wallet.createRandom().address,
      ];
      
      for (const verifier of additionalVerifiers) {
        console.log(`👤 Adding verifier: ${verifier}`);
        const tx = await esgOracle.addAuthorizedVerifier(verifier);
        await tx.wait();
      }
    }
    
    // Fund the contract with some ETH for verification rewards
    const fundAmount = config.verificationReward.mul(100); // Fund for 100 verifications
    console.log(`\n💰 Funding contract with ${ethers.utils.formatEther(fundAmount)} ETH for verification rewards`);
    const fundTx = await deployer.sendTransaction({
      to: esgOracle.address,
      value: fundAmount,
    });
    await fundTx.wait();
    
    // Get final contract stats
    console.log("\n📊 Final contract statistics:");
    const stats = await esgOracle.getContractStats();
    console.log(`   Total Submissions: ${stats._totalSubmissions}`);
    console.log(`   Total Verifications: ${stats._totalVerifications}`);
    console.log(`   Total Data Sources: ${stats._totalDataSources}`);
    console.log(`   Total Verifiers: ${stats._totalVerifiers}`);
    console.log(`   Verification Threshold: ${stats._verificationThreshold}%`);
    
    // Save deployment information
    const deploymentInfo = {
      network: networkName,
      chainId: network.chainId,
      contractAddress: esgOracle.address,
      deploymentTransaction: esgOracle.deployTransaction.hash,
      deployer: deployer.address,
      deploymentTime: new Date().toISOString(),
      configuration: {
        verificationThreshold: config.verificationThreshold,
        minimumVerifiers: config.minimumVerifiers,
        submissionFee: ethers.utils.formatEther(config.submissionFee),
        verificationReward: ethers.utils.formatEther(config.verificationReward),
      },
      dataSources: dataSources.length,
      contractBalance: ethers.utils.formatEther(await ethers.provider.getBalance(esgOracle.address)),
    };
    
    // Save to file
    const deploymentsDir = path.join(__dirname, "..", "deployments");
    if (!fs.existsSync(deploymentsDir)) {
      fs.mkdirSync(deploymentsDir, { recursive: true });
    }
    
    const deploymentFile = path.join(deploymentsDir, `${networkName}.json`);
    fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));
    
    console.log(`\n💾 Deployment information saved to: ${deploymentFile}`);
    
    // Generate ABI file for frontend
    const abiDir = path.join(__dirname, "..", "abi");
    if (!fs.existsSync(abiDir)) {
      fs.mkdirSync(abiDir, { recursive: true });
    }
    
    const artifactPath = path.join(__dirname, "..", "artifacts", "contracts", "ESGDataOracle.sol", "ESGDataOracle.json");
    if (fs.existsSync(artifactPath)) {
      const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
      const abiFile = path.join(abiDir, "ESGDataOracle.json");
      fs.writeFileSync(abiFile, JSON.stringify({
        contractName: "ESGDataOracle",
        abi: artifact.abi,
        bytecode: artifact.bytecode,
        deployedBytecode: artifact.deployedBytecode,
        networks: {
          [network.chainId]: {
            address: esgOracle.address,
            transactionHash: esgOracle.deployTransaction.hash,
          },
        },
      }, null, 2));
      
      console.log(`📄 ABI file generated: ${abiFile}`);
    }
    
    console.log("\n🎉 Deployment completed successfully!");
    console.log(`\n📋 Summary:`);
    console.log(`   Contract Address: ${esgOracle.address}`);
    console.log(`   Network: ${networkName}`);
    console.log(`   Chain ID: ${network.chainId}`);
    console.log(`   Deployer: ${deployer.address}`);
    console.log(`   Transaction: ${esgOracle.deployTransaction.hash}`);
    
    if (networkName !== "hardhat" && networkName !== "ganache") {
      console.log(`\n🔍 Verify contract on Etherscan:`);
      console.log(`   npx hardhat verify --network ${networkName} ${esgOracle.address}`);
    }
    
    return {
      esgOracle,
      deploymentInfo,
    };
    
  } catch (error) {
    console.error("❌ Deployment failed:", error);
    throw error;
  }
}

// Execute deployment if this script is run directly
if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = main;