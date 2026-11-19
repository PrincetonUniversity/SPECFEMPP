.. _jenkins_setup:

Jenkins setup and migration from the GitHub Pull Request Builder
================================================================

This guide sets up the Jenkins jobs that build every pull request and report
the result to GitHub. It exists because the **GitHub Pull Request Builder**
plugin is deprecated and no longer maintained. Its successor is the **GitHub
Branch Source** plugin, used through **Multibranch Pipeline** jobs and a
**GitHub App**. For what the jobs test, see :ref:`tests`.

Placeholders used below: ``<jenkins>`` is the Jenkins server address,
``<org>/<repo>`` is the GitHub repository.

What changes
------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * -
     - Pull Request Builder (old)
     - Branch Source (new)
   * - Jenkins job
     - One job shared by all pull requests
     - One job that creates a sub-job per pull request (``PR-123``)
   * - Login to GitHub
     - A user's token
     - A GitHub App
   * - Result on the pull request
     - Commit status
     - Entry in the "Checks" tab, with a "Re-run" button
   * - Webhook address
     - ``<jenkins>/ghprbhook/``
     - ``<jenkins>/github-webhook/``
   * - Approval of outside pull requests
     - Built in (admin list, "ok to test")
     - Trust setting plus a build strategy (see step 3)
   * - Pull request number in the pipeline
     - ``ghprbPullId``
     - ``CHANGE_ID``
   * - Target branch in the pipeline
     - ``ghprbTargetBranch``
     - ``CHANGE_TARGET``

SPECFEM++ configuration
-----------------------

The values below are what the SPECFEM++ jobs use.

- Jenkins server: ``jenkins.princeton.edu``, folder ``SpecFEM_KOKKOS``.
- GitHub App: "Princeton Jenkins CI".
- Repository: ``PrincetonUniversity/SPECFEMPP``.
- Only pull requests targeting ``main`` or ``devel`` are built. The filter uses
  these patterns:

  .. code-block:: text

      Target-branch regex:  ^(main|devel)$
      Tag regex:            (?!.*)

- Draft pull requests are ignored.
- Trust setting for forks: "Collaborators". Pull requests from branches in the
  main repository build automatically. Pull requests from forks build
  automatically only if the fork's owner is a collaborator on the repository
  (any role, including Read). Other fork pull requests wait until a maintainer
  comments (see step 3).

There is one Multibranch Pipeline job per pipeline file:

.. list-table::
   :header-rows: 1
   :widths: 15 15 45 25

   * - Jenkins job
     - Check name
     - Pipeline file
     - Comment keyword
   * - GNU
     - GCC
     - :repo-file:`.jenkins/gnu_compiler_checks.gvy`
     - ``gcc``
   * - Intel
     - Intel
     - :repo-file:`.jenkins/intel_compiler_checks.gvy`
     - ``intel``
   * - MPI
     - MPI
     - :repo-file:`.jenkins/mpi_compiler_checks.gvy`
     - ``mpi``
   * - Nvidia
     - Nvidia
     - :repo-file:`.jenkins/nvidia_compiler_checks.gvy`
     - ``nvidia``
   * - Trilinos
     - Trilinos
     - :repo-file:`.jenkins/trilinos_compiler_checks.gvy`
     - ``trilinos``

1. Prerequisites
----------------

Access you need
+++++++++++++++

- **Jenkins**: permission to create jobs and add credentials in your folder.
  Installing plugins needs a Jenkins administrator.
- **GitHub**: admin rights on the repository. Installing a GitHub App on an
  organization's repository may additionally need an organization owner.
- **Network**: GitHub must be able to reach ``https://<jenkins>/github-webhook/``.
  If Jenkins is behind a firewall, the firewall must allow **all** of GitHub's
  webhook address ranges (the ``hooks`` list at https://api.github.com/meta).

Jenkins plugins
+++++++++++++++

Ask a Jenkins administrator to install the missing ones.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Plugin
     - Plugin ID
     - Needed for
   * - GitHub Branch Source
     - ``github-branch-source``
     - The multibranch jobs and the "GitHub App" credential (required)
   * - GitHub Checks
     - ``github-checks``
     - Results in the "Checks" tab, a name per job, the "Re-run" button
       (required)
   * - SCM Filter Branch PR
     - ``scm-filter-branch-pr``
     - Building only pull requests that target certain branches
   * - Basic Branch Build Strategies
     - ``basic-branch-build-strategies``
     - Holding back pull requests from untrusted forks
   * - GitHub Pull Request Comment Build
     - ``github-pr-comment-build``
     - Starting a build with a comment

You can tell whether a plugin is installed from the job configuration page:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - If you see this
     - This plugin is installed
   * - "Status Checks Properties" in the Behaviors "Add" list
     - GitHub Checks
   * - "Filter by name including PRs destined for this branch" in the same list
     - SCM Filter Branch PR
   * - A "Build strategies" section below "Property strategy"
     - Basic Branch Build Strategies
   * - "Trigger build on pull request comment" under "Add property"
     - GitHub Pull Request Comment Build

2. Create and configure the GitHub App
--------------------------------------

Each Jenkins server needs its own App. You cannot reuse someone else's, because
the App is tied to your server by its private key and webhook address. Before
creating one, ask the Jenkins administrators whether an App for this server
already exists.

Create the App
++++++++++++++

A GitHub App belongs to a user or an organization, not to a repository. If you
are not an organization owner, create it under your own account: your GitHub
Settings, Developer settings, GitHub Apps, New GitHub App.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - Name
     - Must be unique across GitHub. It is shown next to every check, so name
       the server and purpose, for example "Princeton Jenkins CI".
   * - Homepage URL
     - ``https://<jenkins>/``
   * - Identifying and authorizing users
     - Leave everything as it is
   * - Setup URL
     - Leave empty
   * - Webhook, Active
     - Ticked
   * - Webhook URL
     - ``https://<jenkins>/github-webhook/`` (the trailing slash is required)
   * - Webhook secret
     - Empty, unless the Jenkins administrators have configured one
   * - Where can this App be installed
     - **Any account** (otherwise it cannot be installed on an organization's
       repository)

**Repository permissions** (leave all others at "No access", and do not select
any organization permissions):

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Permission
     - Access
     - Why
   * - Checks
     - Read and write
     - Publish results to the "Checks" tab
   * - Commit statuses
     - Read and write
     - Publish commit statuses
   * - Contents
     - Read-only
     - Check out the code
   * - Pull requests
     - Read-only
     - Find pull requests
   * - Metadata
     - Read-only
     - Set automatically
   * - Issues
     - Read-only
     - Required, because builds are started by a comment. GitHub delivers pull
       request comments as "Issue comment" events and requires this permission
       for them.

**Subscribe to events**: Check run, Pull request, Push, and Issue comment (the
box appears after Issues is set to Read-only). The "Issue comment" event is
required for the comment trigger.

After creating it
+++++++++++++++++

1. Note the **App ID** on the General page (not the Client ID).
2. Generate a **private key** (General page or the "Credentials, Key pairs"
   tab). A ``.pem`` file downloads. Do not create a client secret.
3. **Install the App**: "Install App", your organization, "Only select
   repositories", your repository. If the button says "Request" instead of
   "Install", an organization owner has to approve it.
4. Optional: upload a logo under "Display information" on the General page.
   Otherwise your profile picture is shown next to the checks.

If you change the App's permissions later, the installation has to **accept the
new permissions** before they take effect (repository Settings, GitHub Apps,
Configure, or an organization owner).

Add the App to Jenkins
++++++++++++++++++++++

1. Convert the key to the format Jenkins accepts:

   .. code-block:: bash

       openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt \
         -in downloaded.pem -out jenkins.pem

2. In your Jenkins folder: Credentials, Add, kind **GitHub App**. Enter the App
   ID, paste the contents of ``jenkins.pem``, and set Owner (under Advanced) to
   your organization.
3. Click "Test Connection". It succeeds once the App is installed.
4. Delete both ``.pem`` files from your machine.

3. Create the Jenkins job
-------------------------

Create one **Multibranch Pipeline** job per pipeline file. Configure the first
one completely, then create the others with New Item, "Copy from", and change
only the script path and the check name.

Branch Sources, Add source, GitHub
++++++++++++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Value
   * - Credentials
     - The GitHub App credential
   * - Repository HTTPS URL
     - ``https://github.com/<org>/<repo>.git``

Behaviors
+++++++++

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Behavior
     - Setting
     - Why
   * - Discover branches
     - Remove it
     - Otherwise every pushed branch without a pull request is built
   * - Discover pull requests from origin
     - "Merging the pull request with the current target branch revision"
     - Builds what the target branch will look like after the merge. This still
       builds when the pull request is opened or updated; nothing is merged on
       GitHub.
   * - Filter by name including PRs destined for this branch (with regular
       expression)
     - The branch and tag patterns below
     - Only pull requests targeting these branches are built
   * - Ignore pull requests marked as drafts
     - Add it
     - Optional
   * - Status Checks Properties
     - A name unique to this job; tick "Skip GitHub Branch Source
       notifications"
     - The name is shown in the "Checks" tab. Without the tick, each build is
       also reported a second time as ``continuous-integration/jenkins/pr-merge``.

Patterns for the "destined for this branch" filter (this builds only pull
requests that target ``main`` or ``devel``, and no tags):

.. code-block:: text

    Regular expression for branches:  ^(main|devel)$
    Regular expression for tags:      (?!.*)

The built-in "Filter by name" behaviors match the pull request's own name
(``PR-123``), not its target branch. Use the "destined for this branch"
variant.

Pull requests from forks
++++++++++++++++++++++++

A pull request from a fork runs the contributor's code on your build machines.
Without the settings below, it is built as soon as it is opened. The Trust
setting alone does not prevent that: it only makes Jenkins take the pipeline
file from the target branch.

To build outside pull requests only after a maintainer approves:

1. **Behaviors**: add "Discover pull requests from forks" with the same merge
   strategy, and choose a Trust setting:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Trust
        - Who builds automatically
      * - From users with Admin or Write permission
        - Fork owners with write access to the repository
      * - Collaborators (the one we use)
        - Everyone added to the repository in any role, including Read. Use
          this to whitelist a person without giving write access. Adding a
          person or team to the repository then also grants CI access.

2. **Build strategies**: Add, "Change requests", and tick:

   - "Ignore change requests flagged as originating from an untrusted source"
     (holds back untrusted pull requests)
   - "Ignore rebuilding merge branches when only the target branch changed"
     (otherwise every open pull request is rebuilt whenever the target moves)

3. **Property strategy**: Add property, "Trigger build on pull request
   comment", so that maintainers can start a held-back build from GitHub. Use
   this pattern, with ``<keyword>`` replaced by the job's own keyword:

   .. code-block:: text

       ^\s*(please\s+)?tests?\s+(this|<keyword>)(\s+please)?\s*[.!]?\s*$

   The GCC job, for example, uses:

   .. code-block:: text

       ^\s*(please\s+)?tests?\s+(this|gcc)(\s+please)?\s*[.!]?\s*$

   Every job accepts ``this``, so "test this please" starts all of them, while
   "test gcc please" starts only the GCC job. The pattern is not case
   sensitive and the comment must contain nothing else. One comment starts
   **one** build; a later push by an untrusted contributor is held back again.
   Only comments from trusted users (collaborators) count.

As an optional alternative that we do not use, the property "Trigger build on
pull request label" starts the jobs once, when the label is added.

Do not enable fork discovery before the build strategy is in place.

Remaining settings
++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Value
   * - Build Configuration
     - Mode "by Jenkinsfile"; Script Path is the pipeline file, for example
       ``.jenkins/gnu_compiler_checks.gvy``. Any file name works.
   * - Scan Repository Triggers
     - "Periodically if not otherwise run", 15 minutes. This catches events the
       webhook missed.
   * - Orphaned Item Strategy
     - "Abort builds" and "Discard old items" ticked, 7 days
   * - Pipeline Libraries
     - Leave empty

The first scan of a new job builds **every** open pull request that contains
the pipeline file. Add the build strategy "Skip initial build on first branch
indexing" to avoid that.

4. The pipeline file
--------------------

Existing declarative pipelines work unchanged unless they use ``ghprb*``
variables (see the table at the top). The job only builds pull requests whose
branch contains the file named in "Script Path".

Add this to each ``pipeline { }`` block to cancel a running build when a new
commit is pushed to the same pull request:

.. code-block:: groovy

    options {
        disableConcurrentBuilds(abortPrevious: true)
    }

Add it only **after** the old Pull Request Builder jobs are disabled. Those
jobs share one job across all pull requests, so there the option would abort
other pull requests' builds. All five SPECFEM++ pipeline files have it.

A minimal pipeline file. Our pipelines call :repo-file:`.jenkins/cell.sh` to
build, test and clean up:

.. code-block:: groovy

    pipeline {
        agent { label '<agent-label>' }
        options {
            disableConcurrentBuilds(abortPrevious: true)
        }
        stages {
            stage('Build') {
                steps { sh '.jenkins/cell.sh build' }
            }
            stage('Test') {
                steps { sh '.jenkins/cell.sh test' }
            }
        }
        post {
            always {
                sh '.jenkins/cell.sh clean'
            }
        }
    }

Test results are published with the ``junit`` step. Keep
``skipPublishingChecks: true`` when several jobs report to the same pull
request. Without it each job also publishes a check named "Tests", and they
overwrite each other:

.. code-block:: groovy

    post {
        always {
            junit testResults: 'junit/*.xml', allowEmptyResults: true,
                  skipPublishingChecks: true
        }
    }

5. Test and validate
--------------------

Work through these in order on one job before copying it.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Step
     - Expected result
     - Where to look if not
   * - Click "Scan Repository Now"
     - The Scan Log lists the open pull requests and schedules builds
     - Scan Log on the job page
   * - Open a pull request build's console log
     - A line like ``[GitHub Checks] GitHub check (name: <your name>,
       status: in_progress) has been published.``
     - End of the console log
   * - Open the pull request on GitHub
     - The "Checks" tab shows "``<App name>`` / ``<check name>``"
     - Common errors below
   * - Push a commit to the pull request
     - A new build starts within seconds, without a manual scan
     - App settings, Advanced, Recent Deliveries: a ``pull_request.synchronize``
       entry with a green tick
   * - Click "Re-run" on the check
     - The job builds again
     - Recent Deliveries: a ``check_run.rerequested`` entry
   * - Comment the trigger phrase
     - A build starts
     - Recent Deliveries: an ``issue_comment.created`` entry
   * - Open a fork pull request from an account with no access
     - Nothing builds until a maintainer comments
     - Build strategies on the job

Then run old and new jobs side by side for a few pull requests, update any
branch protection rules to the new check names, and disable the old jobs.

To cancel many queued builds, use the cross next to each entry in the Jenkins
"Build Queue", or ask an administrator to clear the queue.

6. Common errors
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Symptom
     - Cause
     - Fix
   * - Builds start only sometimes after a push; "Redeliver" works on some tries
       and not others
     - GitHub cannot always reach Jenkins. Recent Deliveries shows "failed to
       connect to host". In our case about half of all deliveries failed
       because the firewall allowed only part of GitHub's address ranges. It
       looked like a broken job, but the old plugin was affected too and hid it
       by polling.
     - Have the firewall allow all ranges in the ``hooks`` list at
       https://api.github.com/meta. Turn on the periodic scan as a fallback.
   * - A failed delivery is redelivered successfully, but nothing builds
     - It was the wrong event. With "Check run" subscribed, the App receives an
       event for every check in the repository, including GitHub Actions jobs,
       and Jenkins ignores those.
     - Redeliver the ``pull_request`` event, or click "Scan Repository Now".
   * - Builds run, but nothing appears in the "Checks" tab
     - The job uses a user token instead of the GitHub App, or the GitHub
       Checks plugin is missing. The console log says ``No suitable checks
       publisher found``.
     - Select the App credential; install the plugin.
   * - The check is named "Jenkins"
     - No check name is set on the job
     - Add "Status Checks Properties" with a name
   * - A new check name does not show up
     - The name is fixed when a build starts
     - Start a new build. The old entry stays on that commit.
   * - Only one job's result is visible
     - Several jobs use the same check name and overwrite each other
     - Give each job its own name
   * - An extra ``continuous-integration/jenkins/pr-merge`` entry
     - The Branch Source plugin also posts an old-style commit status
     - Tick "Skip GitHub Branch Source notifications" while no builds are
       running
   * - "Status Checks Properties" or "Build strategies" is missing from the job
       page
     - The plugin is not installed
     - See the plugin tables in step 1
   * - Comments do not start builds, and no ``issue_comment`` entry appears in
       Recent Deliveries
     - "Issue comment" is not ticked, the Issues permission is missing, or the
       installation has not accepted the new permission
     - Fix the App settings; accept the permissions on the installation; then
       post a **new** comment
   * - The comment event arrives, but no build starts
     - The comment does not match the job's pattern, the commenter is not
       trusted, or the pull request has no sub-job (for example fork discovery
       is off)
     - Check the pattern and the Scan Log
   * - Every pushed branch is built
     - "Discover branches" is enabled
     - Remove it
   * - Pull requests against other feature branches are built
     - No target-branch filter
     - Add the "destined for this branch" filter
   * - A burst of builds right after creating a job
     - The first scan builds every open pull request
     - Cancel them, or use "Skip initial build on first branch indexing"
   * - All open pull requests rebuild when the target branch changes
     - The merge strategy treats a new target commit as a change
     - Tick "Ignore rebuilding merge branches when only the target branch
       changed"
   * - A pull request does not appear in a job
     - Its branch does not contain the pipeline file, it is a draft, or it
       targets a filtered-out branch
     - Rebase the branch; check the filters
   * - Other people's builds are aborted
     - ``disableConcurrentBuilds(abortPrevious: true)`` reached the old shared
       jobs
     - Disable the old jobs, or remove the option until they are disabled
   * - Builds wait in "pending" on a commit forever
     - A status was switched off while its build was running
     - Cosmetic; it clears with the next push
   * - The App name is "already taken"
     - App names are unique across GitHub
     - Choose a more specific name; it can be changed later
   * - Jenkins rejects the private key
     - GitHub's key format is not the one Jenkins expects
     - Convert it with the ``openssl pkcs8`` command in step 2
