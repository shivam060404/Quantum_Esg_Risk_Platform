#!/bin/bash

# ESG Risk Assessment Platform Deployment Script
# This script handles deployment to different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="local"
COMPONENT="all"
ACTION="deploy"
NAMESPACE="esg-platform"
VERBOSE=false
DRY_RUN=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (local, staging, production) [default: local]
    -c, --component COMP     Component to deploy (all, backend, frontend, database, redis) [default: all]
    -a, --action ACTION      Action to perform (deploy, rollback, restart, scale, status) [default: deploy]
    -n, --namespace NS       Kubernetes namespace [default: esg-platform]
    -v, --verbose           Enable verbose output
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

Examples:
    $0 --environment staging --component backend
    $0 --action rollback --component frontend
    $0 --dry-run --verbose
    $0 --action status
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--component)
            COMPONENT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$ENVIRONMENT" =~ ^(local|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

if [[ ! "$COMPONENT" =~ ^(all|backend|frontend|database|redis)$ ]]; then
    print_error "Invalid component: $COMPONENT"
    exit 1
fi

if [[ ! "$ACTION" =~ ^(deploy|rollback|restart|scale|status)$ ]]; then
    print_error "Invalid action: $ACTION"
    exit 1
fi

# Function to execute commands
execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$VERBOSE" == "true" ]]; then
        print_info "Executing: $cmd"
    fi
    
    if [[ "$description" != "" ]]; then
        print_info "$description"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would execute: $cmd"
        return 0
    fi
    
    if eval "$cmd"; then
        if [[ "$VERBOSE" == "true" ]]; then
            print_success "Command completed successfully"
        fi
        return 0
    else
        print_error "Command failed: $cmd"
        return 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed (for local environment)
    if [[ "$ENVIRONMENT" == "local" ]] && ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create namespace if it doesn't exist
ensure_namespace() {
    print_info "Ensuring namespace '$NAMESPACE' exists..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_info "Namespace '$NAMESPACE' already exists"
    else
        execute_command "kubectl create namespace $NAMESPACE" "Creating namespace '$NAMESPACE'"
    fi
}

# Function to deploy components
deploy_component() {
    local comp="$1"
    local manifest_file=""
    
    case $comp in
        "database")
            manifest_file="infrastructure/kubernetes/database-deployment.yaml"
            ;;
        "redis")
            manifest_file="infrastructure/kubernetes/redis-deployment.yaml"
            ;;
        "backend")
            manifest_file="infrastructure/kubernetes/backend-deployment.yaml"
            ;;
        "frontend")
            manifest_file="infrastructure/kubernetes/frontend-deployment.yaml"
            ;;
        *)
            print_error "Unknown component: $comp"
            return 1
            ;;
    esac
    
    if [[ ! -f "$manifest_file" ]]; then
        print_error "Manifest file not found: $manifest_file"
        return 1
    fi
    
    execute_command "kubectl apply -f $manifest_file" "Deploying $comp"
    
    # Wait for deployment to be ready
    local deployment_name="esg-$comp"
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=600s" "Waiting for $comp deployment to be ready"
}

# Function to deploy all components
deploy_all() {
    print_info "Deploying all components to $ENVIRONMENT environment..."
    
    # Apply configurations first
    execute_command "kubectl apply -f infrastructure/kubernetes/configmaps-secrets.yaml" "Applying ConfigMaps and Secrets"
    
    # Deploy in order: database, redis, backend, frontend
    deploy_component "database"
    deploy_component "redis"
    deploy_component "backend"
    deploy_component "frontend"
    
    # Apply services and ingress
    execute_command "kubectl apply -f infrastructure/kubernetes/services-ingress.yaml" "Applying Services and Ingress"
    
    print_success "All components deployed successfully"
}

# Function to rollback deployment
rollback_deployment() {
    local comp="$1"
    local deployment_name="esg-$comp"
    
    print_info "Rolling back $comp deployment..."
    execute_command "kubectl rollout undo deployment/$deployment_name -n $NAMESPACE" "Rolling back $comp"
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=300s" "Waiting for rollback to complete"
}

# Function to restart deployment
restart_deployment() {
    local comp="$1"
    local deployment_name="esg-$comp"
    
    print_info "Restarting $comp deployment..."
    execute_command "kubectl rollout restart deployment/$deployment_name -n $NAMESPACE" "Restarting $comp"
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=300s" "Waiting for restart to complete"
}

# Function to show deployment status
show_status() {
    print_info "Deployment Status for namespace: $NAMESPACE"
    
    echo ""
    print_info "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    echo ""
    print_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    print_info "Services:"
    kubectl get services -n "$NAMESPACE" -o wide
    
    echo ""
    print_info "Ingress:"
    kubectl get ingress -n "$NAMESPACE" -o wide
    
    echo ""
    print_info "Persistent Volumes:"
    kubectl get pv -o wide
    
    echo ""
    print_info "Persistent Volume Claims:"
    kubectl get pvc -n "$NAMESPACE" -o wide
}

# Function to scale deployment
scale_deployment() {
    local comp="$1"
    local replicas="$2"
    local deployment_name="esg-$comp"
    
    if [[ -z "$replicas" ]]; then
        print_error "Number of replicas not specified for scaling"
        return 1
    fi
    
    print_info "Scaling $comp deployment to $replicas replicas..."
    execute_command "kubectl scale deployment/$deployment_name --replicas=$replicas -n $NAMESPACE" "Scaling $comp"
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=300s" "Waiting for scaling to complete"
}

# Main execution
main() {
    print_info "ESG Platform Deployment Script"
    print_info "Environment: $ENVIRONMENT"
    print_info "Component: $COMPONENT"
    print_info "Action: $ACTION"
    print_info "Namespace: $NAMESPACE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Ensure namespace exists (except for status action)
    if [[ "$ACTION" != "status" ]]; then
        ensure_namespace
    fi
    
    # Execute action
    case $ACTION in
        "deploy")
            if [[ "$COMPONENT" == "all" ]]; then
                deploy_all
            else
                deploy_component "$COMPONENT"
            fi
            ;;
        "rollback")
            if [[ "$COMPONENT" == "all" ]]; then
                rollback_deployment "backend"
                rollback_deployment "frontend"
            else
                rollback_deployment "$COMPONENT"
            fi
            ;;
        "restart")
            if [[ "$COMPONENT" == "all" ]]; then
                restart_deployment "database"
                restart_deployment "redis"
                restart_deployment "backend"
                restart_deployment "frontend"
            else
                restart_deployment "$COMPONENT"
            fi
            ;;
        "scale")
            # For scaling, we need additional input - this is a simplified version
            print_warning "Scaling requires manual specification of replica count"
            print_info "Use: kubectl scale deployment/esg-$COMPONENT --replicas=N -n $NAMESPACE"
            ;;
        "status")
            show_status
            ;;
    esac
    
    print_success "Operation completed successfully!"
}

# Run main function
main