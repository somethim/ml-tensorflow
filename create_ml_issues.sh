#!/bin/bash

# Make sure gh CLI is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. Please install it first."
    echo "brew install gh  # For Mac"
    echo "Or visit: https://github.com/cli/cli#installation"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Please authenticate with GitHub first using: gh auth login"
    exit 1
fi

# Create ML project-specific labels
echo "Creating labels..."
gh label create "phase-1" --color "0366d6" --description "Foundation Projects" --force
gh label create "phase-2" --color "fbca04" --description "Intermediate Projects" --force
gh label create "phase-3" --color "d93f0b" --description "Advanced Projects" --force
gh label create "phase-4" --color "0e8a16" --description "Expert Projects" --force
gh label create "infrastructure" --color "006b75" --description "Infrastructure & DevOps" --force
gh label create "mobile" --color "1d76db" --description "Mobile Development" --force
gh label create "web" --color "b60205" --description "Web Development" --force
gh label create "api" --color "5319e7" --description "API Development" --force

# Function to create an issue
create_issue() {
    local title="$1"
    local body="$2"
    local labels="$3"

    echo "Creating issue: $title"
    gh issue create \
        --title "$title" \
        --body "$body" \
        --label "$labels" || echo "Failed to create issue: $title"
}

# Phase 1 Projects
create_issue "Binary Classification: Email Spam Detection" "## Tasks
- [ ] Dataset preparation and preprocessing
- [ ] Basic TensorFlow model implementation
- [ ] Model evaluation and optimization
- [ ] Basic Flask API deployment
- [ ] Integration testing with Postman

## Acceptance Criteria
- Model achieves >90% accuracy
- API responds within 100ms
- Documentation complete
- Tests passing" "phase-1"

create_issue "MNIST Digit Recognition" "## Tasks
- [ ] TensorFlow model development
- [ ] Training and validation
- [ ] Model export and serialization
- [ ] Simple React web interface
- [ ] Basic API integration

## Acceptance Criteria
- Model achieves >95% accuracy
- Web interface responsive
- API documentation complete" "phase-1"

# Phase 2 Projects
create_issue "Image Classification with CNN" "## Tasks
- [ ] Dataset preparation with data augmentation
- [ ] CNN architecture design and implementation
- [ ] Model training and optimization
- [ ] Flask API with model serving
- [ ] React Native mobile app integration
- [ ] Next.js web dashboard
- [ ] Laravel backend integration
- [ ] API documentation and testing

## Acceptance Criteria
- Model achieves >90% accuracy
- Mobile app responds within 200ms
- Web dashboard updates in real-time
- All integrations tested" "phase-2,mobile,web,api"

# Infrastructure Tasks
create_issue "API Development Setup" "## Tasks
- [ ] Flask API architecture setup
- [ ] Laravel API integration
- [ ] Authentication system
- [ ] Rate limiting
- [ ] API documentation
- [ ] Testing suite

## Acceptance Criteria
- All endpoints documented
- Authentication working
- Rate limiting implemented
- Tests passing" "infrastructure,api"

create_issue "Mobile App Infrastructure (React Native)" "## Tasks
- [ ] Project structure setup
- [ ] Camera integration
- [ ] API integration
- [ ] Offline functionality
- [ ] Performance optimization
- [ ] UI/UX implementation

## Acceptance Criteria
- App works offline
- Camera integration smooth
- API calls optimized
- UI/UX guidelines followed" "infrastructure,mobile"

create_issue "Web Development Setup (Next.js)" "## Tasks
- [ ] Project architecture
- [ ] API integration
- [ ] Real-time updates
- [ ] Analytics dashboard
- [ ] Admin interface
- [ ] Performance optimization

## Acceptance Criteria
- Dashboard loads under 2s
- Real-time updates working
- Analytics implemented
- Admin features complete" "infrastructure,web"

create_issue "DevOps & Deployment" "## Tasks
- [ ] CI/CD pipeline setup
- [ ] Docker containerization
- [ ] Kubernetes orchestration
- [ ] Monitoring and logging
- [ ] Scaling strategy
- [ ] Backup systems

## Acceptance Criteria
- Automated deployments working
- Monitoring in place
- Scaling tested
- Backup system verified" "infrastructure"

echo "All issues created successfully!"
