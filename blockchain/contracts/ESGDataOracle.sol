// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/**
 * @title ESGDataOracle
 * @dev Smart contract for verifying and storing ESG data with immutable audit trails
 * @author ESG Platform Team
 */
contract ESGDataOracle is Ownable, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // Events
    event ESGDataSubmitted(
        string indexed companyId,
        bytes32 indexed dataHash,
        uint256 timestamp,
        address submitter,
        uint256 verificationScore
    );
    
    event ESGDataVerified(
        string indexed companyId,
        bytes32 indexed dataHash,
        uint256 timestamp,
        address verifier,
        bool verified
    );
    
    event DataSourceAdded(
        address indexed source,
        string name,
        uint256 credibilityScore
    );
    
    event DataSourceRemoved(
        address indexed source,
        string reason
    );
    
    event VerificationThresholdUpdated(
        uint256 oldThreshold,
        uint256 newThreshold
    );

    // Structs
    struct ESGData {
        string companyId;
        bytes32 dataHash;
        uint256 environmentalScore;
        uint256 socialScore;
        uint256 governanceScore;
        uint256 overallScore;
        uint256 carbonEmissions;
        uint256 timestamp;
        address submitter;
        bool verified;
        uint256 verificationScore;
        string[] dataSources;
        mapping(address => bool) verifierVotes;
        uint256 verifierCount;
    }
    
    struct DataSource {
        string name;
        uint256 credibilityScore; // 0-100
        bool active;
        uint256 submissionCount;
        uint256 verificationCount;
        uint256 registrationTime;
    }
    
    struct VerificationRequest {
        string companyId;
        bytes32 dataHash;
        uint256 timestamp;
        address requester;
        bool completed;
        uint256 consensusScore;
    }

    // State variables
    mapping(string => mapping(bytes32 => ESGData)) public esgDataRecords;
    mapping(string => bytes32[]) public companyDataHashes;
    mapping(address => DataSource) public dataSources;
    mapping(bytes32 => VerificationRequest) public verificationRequests;
    
    address[] public authorizedVerifiers;
    mapping(address => bool) public isAuthorizedVerifier;
    
    uint256 public verificationThreshold = 70; // Minimum score for verification
    uint256 public minimumVerifiers = 3;
    uint256 public verificationReward = 0.001 ether;
    uint256 public submissionFee = 0.01 ether;
    
    uint256 public totalSubmissions;
    uint256 public totalVerifications;
    uint256 public totalDataSources;

    // Modifiers
    modifier onlyAuthorizedVerifier() {
        require(isAuthorizedVerifier[msg.sender], "Not authorized verifier");
        _;
    }
    
    modifier onlyRegisteredSource() {
        require(dataSources[msg.sender].active, "Not registered data source");
        _;
    }
    
    modifier validESGScores(uint256 env, uint256 soc, uint256 gov) {
        require(env <= 100 && soc <= 100 && gov <= 100, "Invalid ESG scores");
        _;
    }

    constructor() {
        // Initialize with contract deployer as first verifier
        authorizedVerifiers.push(msg.sender);
        isAuthorizedVerifier[msg.sender] = true;
    }

    /**
     * @dev Submit ESG data for verification
     * @param companyId Unique identifier for the company
     * @param environmentalScore Environmental score (0-100)
     * @param socialScore Social score (0-100)
     * @param governanceScore Governance score (0-100)
     * @param carbonEmissions Carbon emissions in tCO2e
     * @param dataSources Array of data source identifiers
     */
    function submitESGData(
        string memory companyId,
        uint256 environmentalScore,
        uint256 socialScore,
        uint256 governanceScore,
        uint256 carbonEmissions,
        string[] memory dataSources
    ) 
        external 
        payable 
        nonReentrant 
        whenNotPaused 
        onlyRegisteredSource
        validESGScores(environmentalScore, socialScore, governanceScore)
    {
        require(msg.value >= submissionFee, "Insufficient submission fee");
        require(bytes(companyId).length > 0, "Company ID cannot be empty");
        require(dataSources.length > 0, "At least one data source required");
        
        // Calculate overall ESG score
        uint256 overallScore = (environmentalScore + socialScore + governanceScore) / 3;
        
        // Create data hash
        bytes32 dataHash = keccak256(
            abi.encodePacked(
                companyId,
                environmentalScore,
                socialScore,
                governanceScore,
                carbonEmissions,
                block.timestamp,
                msg.sender
            )
        );
        
        // Store ESG data
        ESGData storage newData = esgDataRecords[companyId][dataHash];
        newData.companyId = companyId;
        newData.dataHash = dataHash;
        newData.environmentalScore = environmentalScore;
        newData.socialScore = socialScore;
        newData.governanceScore = governanceScore;
        newData.overallScore = overallScore;
        newData.carbonEmissions = carbonEmissions;
        newData.timestamp = block.timestamp;
        newData.submitter = msg.sender;
        newData.verified = false;
        newData.verificationScore = 0;
        newData.dataSources = dataSources;
        newData.verifierCount = 0;
        
        // Add to company's data hash array
        companyDataHashes[companyId].push(dataHash);
        
        // Update data source statistics
        dataSources[msg.sender].submissionCount++;
        totalSubmissions++;
        
        emit ESGDataSubmitted(
            companyId,
            dataHash,
            block.timestamp,
            msg.sender,
            0
        );
    }

    /**
     * @dev Verify submitted ESG data
     * @param companyId Company identifier
     * @param dataHash Hash of the data to verify
     * @param verified Whether the data is verified
     */
    function verifyESGData(
        string memory companyId,
        bytes32 dataHash,
        bool verified
    ) 
        external 
        nonReentrant 
        whenNotPaused 
        onlyAuthorizedVerifier 
    {
        ESGData storage data = esgDataRecords[companyId][dataHash];
        require(data.timestamp > 0, "Data does not exist");
        require(!data.verifierVotes[msg.sender], "Already voted");
        require(!data.verified, "Already verified");
        
        // Record verifier vote
        data.verifierVotes[msg.sender] = true;
        data.verifierCount++;
        
        if (verified) {
            data.verificationScore += dataSources[data.submitter].credibilityScore;
        }
        
        // Check if verification threshold is met
        if (data.verifierCount >= minimumVerifiers) {
            uint256 averageScore = data.verificationScore / data.verifierCount;
            
            if (averageScore >= verificationThreshold) {
                data.verified = true;
                dataSources[data.submitter].verificationCount++;
                totalVerifications++;
                
                // Reward verifier
                if (address(this).balance >= verificationReward) {
                    payable(msg.sender).transfer(verificationReward);
                }
            }
        }
        
        emit ESGDataVerified(
            companyId,
            dataHash,
            block.timestamp,
            msg.sender,
            verified
        );
    }

    /**
     * @dev Register a new data source
     * @param sourceAddress Address of the data source
     * @param name Name of the data source
     * @param credibilityScore Initial credibility score (0-100)
     */
    function registerDataSource(
        address sourceAddress,
        string memory name,
        uint256 credibilityScore
    ) 
        external 
        onlyOwner 
    {
        require(sourceAddress != address(0), "Invalid address");
        require(bytes(name).length > 0, "Name cannot be empty");
        require(credibilityScore <= 100, "Invalid credibility score");
        require(!dataSources[sourceAddress].active, "Source already registered");
        
        dataSources[sourceAddress] = DataSource({
            name: name,
            credibilityScore: credibilityScore,
            active: true,
            submissionCount: 0,
            verificationCount: 0,
            registrationTime: block.timestamp
        });
        
        totalDataSources++;
        
        emit DataSourceAdded(sourceAddress, name, credibilityScore);
    }

    /**
     * @dev Remove a data source
     * @param sourceAddress Address of the data source to remove
     * @param reason Reason for removal
     */
    function removeDataSource(
        address sourceAddress,
        string memory reason
    ) 
        external 
        onlyOwner 
    {
        require(dataSources[sourceAddress].active, "Source not active");
        
        dataSources[sourceAddress].active = false;
        totalDataSources--;
        
        emit DataSourceRemoved(sourceAddress, reason);
    }

    /**
     * @dev Add authorized verifier
     * @param verifier Address of the verifier to add
     */
    function addAuthorizedVerifier(address verifier) external onlyOwner {
        require(verifier != address(0), "Invalid address");
        require(!isAuthorizedVerifier[verifier], "Already authorized");
        
        authorizedVerifiers.push(verifier);
        isAuthorizedVerifier[verifier] = true;
    }

    /**
     * @dev Remove authorized verifier
     * @param verifier Address of the verifier to remove
     */
    function removeAuthorizedVerifier(address verifier) external onlyOwner {
        require(isAuthorizedVerifier[verifier], "Not authorized verifier");
        
        isAuthorizedVerifier[verifier] = false;
        
        // Remove from array
        for (uint256 i = 0; i < authorizedVerifiers.length; i++) {
            if (authorizedVerifiers[i] == verifier) {
                authorizedVerifiers[i] = authorizedVerifiers[authorizedVerifiers.length - 1];
                authorizedVerifiers.pop();
                break;
            }
        }
    }

    /**
     * @dev Update verification threshold
     * @param newThreshold New threshold value (0-100)
     */
    function updateVerificationThreshold(uint256 newThreshold) external onlyOwner {
        require(newThreshold <= 100, "Invalid threshold");
        
        uint256 oldThreshold = verificationThreshold;
        verificationThreshold = newThreshold;
        
        emit VerificationThresholdUpdated(oldThreshold, newThreshold);
    }

    /**
     * @dev Get ESG data for a company
     * @param companyId Company identifier
     * @param dataHash Hash of the specific data record
     * @return ESG data details
     */
    function getESGData(string memory companyId, bytes32 dataHash)
        external
        view
        returns (
            uint256 environmentalScore,
            uint256 socialScore,
            uint256 governanceScore,
            uint256 overallScore,
            uint256 carbonEmissions,
            uint256 timestamp,
            address submitter,
            bool verified,
            uint256 verificationScore
        )
    {
        ESGData storage data = esgDataRecords[companyId][dataHash];
        return (
            data.environmentalScore,
            data.socialScore,
            data.governanceScore,
            data.overallScore,
            data.carbonEmissions,
            data.timestamp,
            data.submitter,
            data.verified,
            data.verificationScore
        );
    }

    /**
     * @dev Get latest verified ESG data for a company
     * @param companyId Company identifier
     * @return Latest verified ESG data
     */
    function getLatestVerifiedESGData(string memory companyId)
        external
        view
        returns (
            bytes32 dataHash,
            uint256 environmentalScore,
            uint256 socialScore,
            uint256 governanceScore,
            uint256 overallScore,
            uint256 timestamp,
            bool verified
        )
    {
        bytes32[] memory hashes = companyDataHashes[companyId];
        
        // Find latest verified data
        for (int256 i = int256(hashes.length) - 1; i >= 0; i--) {
            ESGData storage data = esgDataRecords[companyId][hashes[uint256(i)]];
            if (data.verified) {
                return (
                    data.dataHash,
                    data.environmentalScore,
                    data.socialScore,
                    data.governanceScore,
                    data.overallScore,
                    data.timestamp,
                    data.verified
                );
            }
        }
        
        // Return empty if no verified data found
        return (bytes32(0), 0, 0, 0, 0, 0, false);
    }

    /**
     * @dev Get data source information
     * @param sourceAddress Address of the data source
     * @return Data source details
     */
    function getDataSource(address sourceAddress)
        external
        view
        returns (
            string memory name,
            uint256 credibilityScore,
            bool active,
            uint256 submissionCount,
            uint256 verificationCount,
            uint256 registrationTime
        )
    {
        DataSource storage source = dataSources[sourceAddress];
        return (
            source.name,
            source.credibilityScore,
            source.active,
            source.submissionCount,
            source.verificationCount,
            source.registrationTime
        );
    }

    /**
     * @dev Get contract statistics
     * @return Contract usage statistics
     */
    function getContractStats()
        external
        view
        returns (
            uint256 _totalSubmissions,
            uint256 _totalVerifications,
            uint256 _totalDataSources,
            uint256 _totalVerifiers,
            uint256 _verificationThreshold
        )
    {
        return (
            totalSubmissions,
            totalVerifications,
            totalDataSources,
            authorizedVerifiers.length,
            verificationThreshold
        );
    }

    /**
     * @dev Get all data hashes for a company
     * @param companyId Company identifier
     * @return Array of data hashes
     */
    function getCompanyDataHashes(string memory companyId)
        external
        view
        returns (bytes32[] memory)
    {
        return companyDataHashes[companyId];
    }

    /**
     * @dev Withdraw contract balance (only owner)
     */
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        payable(owner()).transfer(balance);
    }

    /**
     * @dev Update submission fee
     * @param newFee New submission fee in wei
     */
    function updateSubmissionFee(uint256 newFee) external onlyOwner {
        submissionFee = newFee;
    }

    /**
     * @dev Update verification reward
     * @param newReward New verification reward in wei
     */
    function updateVerificationReward(uint256 newReward) external onlyOwner {
        verificationReward = newReward;
    }

    /**
     * @dev Pause contract (emergency)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause contract
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Receive function to accept Ether
     */
    receive() external payable {}

    /**
     * @dev Fallback function
     */
    fallback() external payable {}
}