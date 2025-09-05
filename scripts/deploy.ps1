# ESG Risk Assessment Platform Deployment Script (PowerShell)
# This script handles deployment to different environments on Windows

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("local", "staging", "production")]
    [string]$Environment = "local",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("all", "backend", "frontend", "database", "redis")]
    [string]$Component = "all",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("deploy", "rollback", "restart", "scale", "status")]
    [string]$Action = "deploy",
    
    [Parameter(Mandatory=$false)]
    [string]$Namespace = "esg-platform",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
}

# Function to print colored output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Function to show usage
function Show-Usage {
    Write-Host @"
Usage: .\deploy.ps1 [OPTIONS]

Parameters:
    -Environment ENV     Target environment (local, staging, production) [default: local]
    -Component COMP      Component to deploy (all, backend, frontend, database, redis) [default: all]
    -Action ACTION       Action to perform (deploy, rollback, restart, scale, status) [default: deploy]
    -Namespace NS        Kubernetes namespace [default: esg-platform]
    -Verbose            Enable verbose output
    -DryRun             Show what would be done without executing
    -Help               Show this help message

Examples:
    .\deploy.ps1 -Environment staging -Component backend
    .\deploy.ps1 -Action rollback -Component frontend
    .\deploy.ps1 -DryRun -Verbose
    .\deploy.ps1 -Action status
"@
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Function to execute commands
function Invoke-Command {
    param(
        [string]$Command,
        [string]$Description = ""
    )
    
    if ($Verbose) {
        Write-Info "Executing: $Command"
    }
    
    if ($Description -ne "") {
        Write-Info $Description
    }
    
    if ($DryRun) {
        Write-Host "[DRY-RUN] Would execute: $Command" -ForegroundColor $Colors.Cyan
        return $true
    }
    
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0) {
            if ($Verbose) {
                Write-Success "Command completed successfully"
            }
            return $true
        } else {
            Write-Error "Command failed with exit code $LASTEXITCODE: $Command"
            return $false
        }
    } catch {
        Write-Error "Command failed: $Command - $($_.Exception.Message)"
        return $false
    }
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if kubectl is installed
    try {
        $null = Get-Command kubectl -ErrorAction Stop
    } catch {
        Write-Error "kubectl is not installed or not in PATH"
        exit 1
    }
    
    # Check if docker is installed (for local environment)
    if ($Environment -eq "local") {
        try {
            $null = Get-Command docker -ErrorAction Stop
        } catch {
            Write-Error "Docker is not installed or not in PATH"
            exit 1
        }
    }
    
    # Check if we can connect to Kubernetes cluster
    try {
        kubectl cluster-info | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Cannot connect to cluster"
        }
    } catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Function to create namespace if it doesn't exist
function Ensure-Namespace {
    Write-Info "Ensuring namespace '$Namespace' exists..."
    
    try {
        kubectl get namespace $Namespace | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Namespace '$Namespace' already exists"
        } else {
            throw "Namespace does not exist"
        }
    } catch {
        $result = Invoke-Command "kubectl create namespace $Namespace" "Creating namespace '$Namespace'"
        if (-not $result) {
            exit 1
        }
    }
}

# Function to deploy components
function Deploy-Component {
    param([string]$ComponentName)
    
    $manifestFile = ""
    
    switch ($ComponentName) {
        "database" { $manifestFile = "infrastructure/kubernetes/database-deployment.yaml" }
        "redis" { $manifestFile = "infrastructure/kubernetes/redis-deployment.yaml" }
        "backend" { $manifestFile = "infrastructure/kubernetes/backend-deployment.yaml" }
        "frontend" { $manifestFile = "infrastructure/kubernetes/frontend-deployment.yaml" }
        default {
            Write-Error "Unknown component: $ComponentName"
            return $false
        }
    }
    
    if (-not (Test-Path $manifestFile)) {
        Write-Error "Manifest file not found: $manifestFile"
        return $false
    }
    
    $result1 = Invoke-Command "kubectl apply -f $manifestFile" "Deploying $ComponentName"
    if (-not $result1) {
        return $false
    }
    
    # Wait for deployment to be ready
    $deploymentName = "esg-$ComponentName"
    $result2 = Invoke-Command "kubectl rollout status deployment/$deploymentName -n $Namespace --timeout=600s" "Waiting for $ComponentName deployment to be ready"
    return $result2
}

# Function to deploy all components
function Deploy-All {
    Write-Info "Deploying all components to $Environment environment..."
    
    # Apply configurations first
    $result = Invoke-Command "kubectl apply -f infrastructure/kubernetes/configmaps-secrets.yaml" "Applying ConfigMaps and Secrets"
    if (-not $result) {
        return $false
    }
    
    # Deploy in order: database, redis, backend, frontend
    $components = @("database", "redis", "backend", "frontend")
    foreach ($comp in $components) {
        $result = Deploy-Component $comp
        if (-not $result) {
            return $false
        }
    }
    
    # Apply services and ingress
    $result = Invoke-Command "kubectl apply -f infrastructure/kubernetes/services-ingress.yaml" "Applying Services and Ingress"
    if (-not $result) {
        return $false
    }
    
    Write-Success "All components deployed successfully"
    return $true
}

# Function to rollback deployment
function Rollback-Deployment {
    param([string]$ComponentName)
    
    $deploymentName = "esg-$ComponentName"
    
    Write-Info "Rolling back $ComponentName deployment..."
    $result1 = Invoke-Command "kubectl rollout undo deployment/$deploymentName -n $Namespace" "Rolling back $ComponentName"
    if (-not $result1) {
        return $false
    }
    
    $result2 = Invoke-Command "kubectl rollout status deployment/$deploymentName -n $Namespace --timeout=300s" "Waiting for rollback to complete"
    return $result2
}

# Function to restart deployment
function Restart-Deployment {
    param([string]$ComponentName)
    
    $deploymentName = "esg-$ComponentName"
    
    Write-Info "Restarting $ComponentName deployment..."
    $result1 = Invoke-Command "kubectl rollout restart deployment/$deploymentName -n $Namespace" "Restarting $ComponentName"
    if (-not $result1) {
        return $false
    }
    
    $result2 = Invoke-Command "kubectl rollout status deployment/$deploymentName -n $Namespace --timeout=300s" "Waiting for restart to complete"
    return $result2
}

# Function to show deployment status
function Show-Status {
    Write-Info "Deployment Status for namespace: $Namespace"
    
    Write-Host ""
    Write-Info "Deployments:"
    kubectl get deployments -n $Namespace -o wide
    
    Write-Host ""
    Write-Info "Pods:"
    kubectl get pods -n $Namespace -o wide
    
    Write-Host ""
    Write-Info "Services:"
    kubectl get services -n $Namespace -o wide
    
    Write-Host ""
    Write-Info "Ingress:"
    kubectl get ingress -n $Namespace -o wide
    
    Write-Host ""
    Write-Info "Persistent Volumes:"
    kubectl get pv -o wide
    
    Write-Host ""
    Write-Info "Persistent Volume Claims:"
    kubectl get pvc -n $Namespace -o wide
}

# Function to scale deployment
function Scale-Deployment {
    param(
        [string]$ComponentName,
        [int]$Replicas
    )
    
    if ($Replicas -eq 0) {
        Write-Error "Number of replicas not specified for scaling"
        return $false
    }
    
    $deploymentName = "esg-$ComponentName"
    
    Write-Info "Scaling $ComponentName deployment to $Replicas replicas..."
    $result1 = Invoke-Command "kubectl scale deployment/$deploymentName --replicas=$Replicas -n $Namespace" "Scaling $ComponentName"
    if (-not $result1) {
        return $false
    }
    
    $result2 = Invoke-Command "kubectl rollout status deployment/$deploymentName -n $Namespace --timeout=300s" "Waiting for scaling to complete"
    return $result2
}

# Main execution
function Main {
    Write-Info "ESG Platform Deployment Script (PowerShell)"
    Write-Info "Environment: $Environment"
    Write-Info "Component: $Component"
    Write-Info "Action: $Action"
    Write-Info "Namespace: $Namespace"
    
    if ($DryRun) {
        Write-Warning "DRY RUN MODE - No actual changes will be made"
    }
    
    Write-Host ""
    
    # Check prerequisites
    Test-Prerequisites
    
    # Ensure namespace exists (except for status action)
    if ($Action -ne "status") {
        Ensure-Namespace
    }
    
    # Execute action
    $success = $true
    
    switch ($Action) {
        "deploy" {
            if ($Component -eq "all") {
                $success = Deploy-All
            } else {
                $success = Deploy-Component $Component
            }
        }
        "rollback" {
            if ($Component -eq "all") {
                $success = (Rollback-Deployment "backend") -and (Rollback-Deployment "frontend")
            } else {
                $success = Rollback-Deployment $Component
            }
        }
        "restart" {
            if ($Component -eq "all") {
                $components = @("database", "redis", "backend", "frontend")
                foreach ($comp in $components) {
                    $result = Restart-Deployment $comp
                    $success = $success -and $result
                }
            } else {
                $success = Restart-Deployment $Component
            }
        }
        "scale" {
            # For scaling, we need additional input - this is a simplified version
            Write-Warning "Scaling requires manual specification of replica count"
            Write-Info "Use: kubectl scale deployment/esg-$Component --replicas=N -n $Namespace"
            $success = $true
        }
        "status" {
            Show-Status
            $success = $true
        }
    }
    
    if ($success) {
        Write-Success "Operation completed successfully!"
        exit 0
    } else {
        Write-Error "Operation failed!"
        exit 1
    }
}

# Run main function
Main