Family planning model and analyses
==================================

This repository contains the code for both the family planning model, FPsim, as well as the analysis scripts for performing analyses.

* FPsim, in the folder ``fpsim``, is a standalone Python library for performing family planning analyses.
* Scripts that use FPsim to run analyses are in the ``fp_analyses`` folder.
* FPsim tests are in the ``tests`` folder.
* Most other folders are preserved for archival purposes and may be deleted.
* This repository is currently under development. Eventually, FPsim and ``fp_analyses`` will be split into a separate repositories.

Installation
------------

Run ``pip install -e .`` to install these packages and their required dependencies. This will make both ``fpsim`` and ``fp_analyses`` available on the Python path.


Contributing
------------

* Everything you're working on must be linked to an issue. If you notice that something needs to be done (even small things or things nearly finished) and there isn't an issue for it, create an issue. This helps track who is doing what and why.
* ALL PRs should be linked to at least one issue. As above, if you're working on a PR and there's no issue associated with it, you can create an issue. However, before doing so, ask yourself if it really needs to be done. 
* All PRs should have another person assigned for review. Use your best judgement here, as roles shift, but in general: 
   - Anyone you've worked with on this issue 1:1
   - @MOBrien-IDM as FPsim lead
   - @cliffckerr to ensure new feature performance and compatibility with FPsim
   - @mzimmermann-IDM for subject matter expertise, economic and empowerment questions, questions about modeling best practices
   - @avictorious for questions about historical FPsim decisions and subject matter expertise
   - @SBuxton-IDM or @CWiswell-IDM for testing and debugging
* Keep PRs as small as possible: e.g., one issue, one PR. Small PRs are easier to review and merge. 
* At times there may be a backlog of issues, but there should never be a big backlog of PRs. (If you're unsure whether to make a PR, write a detailed issue first.)
   - What if there are two people working on PRs at the same time?
      - Take a look at the issue priority. The PR addressing the higher priority issue should merge first. Make sure you pull the new master after that merge before you push changes for your PR. If both issues are high priority, the one with more time-sensitive commits should be merged first. If you're unsure, ask. 
* If we do have a backlog of PRs, it's fine to make a new branch off your current PR, and make a new PR from that. These "cumulative PRs are not ideal, but they are better than creating merge conflicts with yourself!
* Before starting work, always ensure you've pulled from master. If you spend more than a few days on your PR, make sure you pull from master regularly. Before making a PR, ensure that your branch is up to date with master.
* Make sure tests pass on your PR. If they don't, mark the PR as draft until they do.
* Even if your work isn't ready for a PR, push it regularly. A guiding principle is to commit every few minutes and push to your branch every 1-2 hours.

